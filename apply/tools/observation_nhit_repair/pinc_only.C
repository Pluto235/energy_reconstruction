#include <TFile.h>
#include <TTree.h>
#include <TLeaf.h>
#include <TObject.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct NhitBenchmarkStats {
  Long64_t raw=0, selected=0, scan_bytes=0, hit_bytes=0, scan_calls=0, hit_calls=0;
  double scan_seconds=0, hit_seconds=0, write_seconds=0;
  Long64_t charge_boundary=0, time_boundary=0, over_4096=0;
};
static bool counted_hit(float q, double t) { return q >= 0.3f && std::abs(t) < 30.; }
void nhit_benchmark_selfcheck() {
  assert(counted_hit(0.3f,0));
  assert(!counted_hit(std::nextafter(0.3f,0.f),0));
  assert(counted_hit(std::nextafter(0.3f,1.f),0));
  assert(!counted_hit(1.f,30.) && !counted_hit(1.f,-30.));
  assert(counted_hit(1.f,std::nextafter(30.,0.)));
  assert(counted_hit(1.f,std::nextafter(-30.,0.)));
}
NhitBenchmarkStats nhit_benchmark_filter(const char *src, const char *dst) {
  using Clock=std::chrono::steady_clock;
  auto seconds=[](Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();};
  std::unique_ptr<TFile> in(TFile::Open(src,"READ"));
  if (!in || in->IsZombie() || in->TestBit(TFile::kRecovered)) throw std::runtime_error("source unreadable/recovered");
  auto t=dynamic_cast<TTree*>(in->Get("t_eventout"));
  if (!t) throw std::runtime_error("missing event tree");
  std::vector<std::string> cols={"irun","ies","iseq","ievent","nv","vqsamp","vt"};
  for(auto &s:cols) if(!t->GetLeaf(s.c_str())) throw std::runtime_error("missing leaf "+s);
  if(!t->GetLeaf("pincness")) throw std::runtime_error("missing PINC");
  if(std::string(t->GetLeaf("vqsamp")->GetTypeName())!="Float_t") throw std::runtime_error("charge precision changed");
  NhitBenchmarkStats s; s.raw=t->GetEntries();
  t->SetBranchStatus("*",0); t->SetBranchStatus("pincness",1);
  t->SetCacheSize(8*1024*1024); t->AddBranchToCache("pincness",true); t->StopCacheLearningPhase();
  std::vector<Long64_t> selected; selected.reserve(s.raw/20+1024);
  auto begin=Clock::now(); auto bytes=in->GetBytesRead(); auto calls=in->GetReadCalls();
  for(Long64_t i=0;i<s.raw;++i) {
    if(t->GetEntry(i)<=0) throw std::runtime_error("PINC read failure");
    if(t->GetLeaf("pincness")->GetValue()<1.1) selected.push_back(i);
  }
  s.scan_seconds=seconds(begin); s.scan_bytes=in->GetBytesRead()-bytes; s.scan_calls=in->GetReadCalls()-calls;
  s.selected=selected.size();
  t->SetCacheSize(0);t->SetBranchStatus("*",0);
  for(auto &c:cols)t->SetBranchStatus(c.c_str(),1);
  t->SetCacheSize(32*1024*1024);t->SetCacheEntryRange(0,s.raw);
  for(auto &c:cols)t->AddBranchToCache(c.c_str(),true);
  t->StopCacheLearningPhase();
  std::unique_ptr<TFile> out(TFile::Open(dst,"CREATE","",104));
  if(!out || out->IsZombie()) throw std::runtime_error("output exists/unwritable");
  TTree result("t_candidate","PINC-only selection; not ETO-aligned");result.SetDirectory(out.get());
  Int_t irun,ies,iseq,ievent,nv,nhit;Long64_t ihep_entry;
  result.Branch("irun",&irun,"irun/I");result.Branch("ies",&ies,"ies/I");
  result.Branch("iseq",&iseq,"iseq/I");result.Branch("ievent",&ievent,"ievent/I");
  result.Branch("nv",&nv,"nv/I");result.Branch("nhit",&nhit,"nhit/I");
  result.Branch("ihep_entry",&ihep_entry,"ihep_entry/L");
  begin=Clock::now();bytes=in->GetBytesRead();calls=in->GetReadCalls();
  for(auto entry:selected) {
    if(t->GetEntry(entry)<=0) throw std::runtime_error("selected hit read failure");
    irun=t->GetLeaf("irun")->GetValue();ies=t->GetLeaf("ies")->GetValue();
    iseq=t->GetLeaf("iseq")->GetValue();ievent=t->GetLeaf("ievent")->GetValue();
    nv=t->GetLeaf("nv")->GetValue();ihep_entry=entry;nhit=0;
    auto q=t->GetLeaf("vqsamp"), vt=t->GetLeaf("vt");
    if(nv<0 || q->GetNdata()!=nv || vt->GetNdata()!=nv) throw std::runtime_error("hit length differs from nv");
    if(nv>4096) ++s.over_4096;
    for(Int_t j=0;j<nv;++j) {
      auto charge=static_cast<float>(q->GetValue(j));auto time=vt->GetValue(j);
      if(!std::isfinite(charge)||!std::isfinite(time)) throw std::runtime_error("nonfinite charge/time");
      s.charge_boundary += (charge==0.3f); s.time_boundary += (std::abs(time)==30.);
      nhit+=counted_hit(charge,time);
    }
    auto write_start=Clock::now();
    if(result.Fill()<0||out->TestBit(TFile::kWriteError)) throw std::runtime_error("event write failed");
    s.write_seconds+=seconds(write_start);
  }
  auto final_write=Clock::now();
  if(result.Write("",TObject::kOverwrite)<=0||out->TestBit(TFile::kWriteError)) throw std::runtime_error("write failed");
  s.write_seconds+=seconds(final_write);
  s.hit_seconds=seconds(begin)-s.write_seconds;s.hit_bytes=in->GetBytesRead()-bytes;s.hit_calls=in->GetReadCalls()-calls;
  result.SetDirectory(nullptr);out->Close();return s;
}
