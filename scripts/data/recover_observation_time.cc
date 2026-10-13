// Reduced lookup and Cog conversion follow liushijie/time-recover/src/recover_wcda_time.cc.
#include <TBranch.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TSystem.h>
#include <TTree.h>
#include <TNamed.h>
#include <TError.h>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <limits>
#include <sys/resource.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <vector>

#include "hpatimer.h"
#include "papi.h"

namespace {

struct Key {
  int irun = 0;
  int ies = 0;
  int iseq = 0;

  bool operator<(const Key &other) const {
    return std::tie(irun, ies, iseq) < std::tie(other.irun, other.ies, other.iseq);
  }
  bool operator==(const Key &other) const {
    return irun == other.irun && ies == other.ies && iseq == other.iseq;
  }
};

struct FileMeta {
  std::string yyyymmdd;
  std::string mmdd;
  std::string hour;
};

struct ReducedContext {
  Key key;
  std::string path;
  std::unique_ptr<TFile> file;
  TTree *tree = nullptr;
  TLeaf *leaf_ievent = nullptr;
  TLeaf *leaf_sec = nullptr;
  TLeaf *leaf_subsec = nullptr;
  int ievent = 0;
  Long64_t sec = 0;
  double subsec = 0.0;
  bool use_leaf_fallback = false;
  bool used_ies_fallback = false;
  Long64_t entries = 0;
  std::vector<int> event_ids;
  std::vector<Long64_t> secs;
  std::vector<double> subsecs;
};

struct MissingReducedCache {
  bool valid = false;
  Key key;

  bool contains(const Key &candidate) const {
    return valid && key == candidate;
  }

  void set(const Key &candidate) {
    valid = true;
    key = candidate;
  }
};

std::string shell_quote(const std::string &s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

bool parse_eval_filename(const std::string &path, FileMeta &meta) {
  static const std::regex re(R"(Esg([0-9]{8})_([0-9]{2})\.root$)");
  std::smatch m;
  const std::string name = gSystem->BaseName(path.c_str());
  if (!std::regex_search(name, m, re)) return false;
  meta.yyyymmdd = m[1].str();
  meta.mmdd = meta.yyyymmdd.substr(4, 4);
  meta.hour = m[2].str();
  return true;
}

std::string join_path(const std::string &a, const std::string &b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

std::string dirname_of(const std::string &path) {
  const std::string::size_type pos = path.find_last_of('/');
  if (pos == std::string::npos) return ".";
  if (pos == 0) return "/";
  return path.substr(0, pos);
}

std::string basename_of(const std::string &path) {
  const std::string::size_type pos = path.find_last_of('/');
  if (pos == std::string::npos) return path;
  return path.substr(pos + 1);
}

bool path_exists(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

std::map<Key, std::string> build_manifest(const std::string &day_dir) {
  std::map<Key, std::string> manifest;
  DIR *dir = opendir(day_dir.c_str());
  if (!dir) return manifest;

  static const std::regex re(
      R"(ES\.([0-9]+)\.WCDA_EVENT\..*\.es-([0-9]+)\.[0-9]{14}\.([0-9]+)\.reduced\.root$)");
  while (dirent *dent = readdir(dir)) {
    const std::string name = dent->d_name;
    std::smatch m;
    if (!std::regex_match(name, m, re)) continue;

    Key key;
    key.irun = std::stoi(m[1].str());
    key.ies = std::stoi(m[2].str());
    key.iseq = std::stoi(m[3].str());

    const std::string path = join_path(day_dir, name);
    auto result = manifest.emplace(key, path);
    auto it = result.first;
    const bool inserted = result.second;
    if (!inserted && path != it->second)
      throw std::runtime_error("Ambiguous reduced file key: " + path + " and " + it->second);
  }
  closedir(dir);
  return manifest;
}

std::map<Key, std::string> read_manifest_file(const std::string &manifest_path,
                                               const FileMeta &meta) {
  std::ifstream is(manifest_path);
  std::map<Key, std::string> manifest;
  if (!is) return manifest;

  std::string line;
  while (std::getline(is, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream ss(line);
    std::string yyyymmdd;
    Key key;
    std::string path;
    if (!(ss >> yyyymmdd >> key.irun >> key.ies >> key.iseq >> path)) continue;
    if (yyyymmdd != meta.yyyymmdd) continue;

    auto result = manifest.emplace(key, path);
    auto it = result.first;
    const bool inserted = result.second;
    if (!inserted && path != it->second)
      throw std::runtime_error("Ambiguous reduced file key: " + path + " and " + it->second);
  }
  return manifest;
}

std::map<std::pair<int, int>, std::string> build_unique_run_seq_manifest(
    const std::map<Key, std::string> &manifest) {
  std::map<std::pair<int, int>, std::string> unique;
  std::set<std::pair<int, int>> ambiguous;

  for (const auto &item : manifest) {
    const Key &key = item.first;
    const std::pair<int, int> run_seq(key.irun, key.iseq);
    if (ambiguous.count(run_seq)) continue;

    auto it = unique.find(run_seq);
    if (it == unique.end()) {
      unique.emplace(run_seq, item.second);
    } else if (it->second != item.second) {
      unique.erase(it);
      ambiguous.insert(run_seq);
    }
  }
  return unique;
}

std::unique_ptr<TFile> open_root_with_retries(const std::string &path, int attempts = 3) {
  for (int attempt = 1; attempt <= attempts; ++attempt) {
    std::unique_ptr<TFile> file(TFile::Open(path.c_str(), "READ"));
    if (file && !file->IsZombie()) return file;
    file.reset();
    if (attempt < attempts) gSystem->Sleep(1000 * attempt);
  }
  return nullptr;
}

bool setup_reduced_branches(ReducedContext &ctx) {
  ctx.tree->SetBranchStatus("*", 0);
  const bool has_branches =
      ctx.tree->GetBranch("ievent") && ctx.tree->GetBranch("sec") &&
      ctx.tree->GetBranch("subsec");
  if (has_branches) {
    ctx.tree->SetBranchStatus("ievent", 1);
    ctx.tree->SetBranchStatus("sec", 1);
    ctx.tree->SetBranchStatus("subsec", 1);
    if (ctx.tree->SetBranchAddress("ievent", &ctx.ievent) >= 0 &&
        ctx.tree->SetBranchAddress("sec", &ctx.sec) >= 0 &&
        ctx.tree->SetBranchAddress("subsec", &ctx.subsec) >= 0) {
      ctx.use_leaf_fallback = false;
      return true;
    }
  }

  ctx.tree->SetBranchStatus("*", 0);
  ctx.tree->SetBranchStatus("ievent", 1);
  ctx.tree->SetBranchStatus("sec", 1);
  ctx.tree->SetBranchStatus("subsec", 1);
  ctx.leaf_ievent = ctx.tree->GetLeaf("ievent");
  ctx.leaf_sec = ctx.tree->GetLeaf("sec");
  ctx.leaf_subsec = ctx.tree->GetLeaf("subsec");
  ctx.use_leaf_fallback = true;
  return ctx.leaf_ievent && ctx.leaf_sec && ctx.leaf_subsec;
}

ReducedContext *load_reduced(
    const Key &key, const std::map<Key, std::string> &manifest,
    const std::map<std::pair<int, int>, std::string> &run_seq_manifest,
    ReducedContext &ctx, MissingReducedCache &missing_cache, bool &used_ies_fallback) {
  if (ctx.file && ctx.key == key) {
    used_ies_fallback = ctx.used_ies_fallback;
    return &ctx;
  }
  if (missing_cache.contains(key)) return nullptr;

  ctx = ReducedContext{};
  ctx.key = key;
  used_ies_fallback = false;

  auto it = manifest.find(key);
  if (it == manifest.end()) {
    const std::pair<int, int> run_seq(key.irun, key.iseq);
    auto fallback = run_seq_manifest.find(run_seq);
    if (fallback == run_seq_manifest.end()) {
      missing_cache.set(key);
      return nullptr;
    }
    ctx.path = fallback->second;
    used_ies_fallback = true;
  } else {
    ctx.path = it->second;
  }
  ctx.used_ies_fallback = used_ies_fallback;
  ctx.file = open_root_with_retries(ctx.path);
  if (!ctx.file) {
    missing_cache.set(key);
    return nullptr;
  }

  ctx.tree = dynamic_cast<TTree *>(ctx.file->Get("wcda"));
  if (!ctx.tree) {
    ctx.file.reset();
    missing_cache.set(key);
    return nullptr;
  }
  ctx.entries = ctx.tree->GetEntries();
  if (!setup_reduced_branches(ctx)) {
    ctx.tree = nullptr;
    return &ctx;
  }

  if(ctx.entries<0 || ctx.entries>512LL*1024*1024/20) throw std::runtime_error("Reduced scalar cache exceeds 512 MiB");
  ctx.event_ids.resize(ctx.entries);
  ctx.secs.resize(ctx.entries);
  ctx.subsecs.resize(ctx.entries);
  for (Long64_t i = 0; i < ctx.entries; ++i) {
    if (ctx.tree->GetEntry(i) <= 0) throw std::runtime_error("Reduced read failed: " + ctx.path);
    if (ctx.use_leaf_fallback) {
      ctx.event_ids[i] = static_cast<int>(ctx.leaf_ievent->GetValue());
      ctx.secs[i] = static_cast<Long64_t>(ctx.leaf_sec->GetValueLong64());
      ctx.subsecs[i] = ctx.leaf_subsec->GetValue();
    } else {
      ctx.event_ids[i] = ctx.ievent;
      ctx.secs[i] = ctx.sec;
      ctx.subsecs[i] = ctx.subsec;
    }
  }
  return &ctx;
}


bool root_error = false;
void error_handler(int level, Bool_t abort, const char *location, const char *message) {
  if (level >= kError) root_error = true;
  DefaultErrorHandler(level, abort, location, message);
}
void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
bool same_double(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
using EventKey = std::tuple<int,int,int,int>;
struct Event {
  int irun=0, ies=0, iseq=0, ievent=0;
  double theta=0, phi=0;
  EventKey key() const { return {irun,ies,iseq,ievent}; }
};
struct TimeRow {
  Long64_t tai_sec=0, unix_sec=0;
  double tai_subsec=0, unix_subsec=0, mjd=0;
  int mjdns4=0;
  double ra_mean=0, dec_mean=0, ra_mean_deg=0, dec_mean_deg=0;
  int match_status=0;
};

template<class T> void field(TTree *tree,const char *name,T *value,const char *type,bool writing) {
  if(writing) {
    std::string leaf=std::string(name)+"/"+type;
    require(tree->Branch(name,value,leaf.c_str())!=nullptr,"Cannot create branch "+std::string(name));
  } else {
    auto branch=tree->GetBranch(name); auto leaf=tree->GetLeaf(name);
    std::string expected=std::string(type)=="L"?"Long64_t":std::string(type)=="I"?"Int_t":"Double_t";
    require(branch && leaf && expected==leaf->GetTypeName(),"Missing or wrong branch type: "+std::string(name));
    require(branch->GetEntries()==tree->GetEntries(),"Incomplete branch: "+std::string(name));
    tree->SetBranchStatus(name,1);
    require(tree->SetBranchAddress(name,value)>=0,"Cannot bind branch "+std::string(name));
  }
}
void event_fields(TTree *t,Event &e,bool w) {
  field(t,"irun",&e.irun,"I",w);field(t,"ies",&e.ies,"I",w);
  field(t,"iseq",&e.iseq,"I",w);field(t,"ievent",&e.ievent,"I",w);
  field(t,"theta",&e.theta,"D",w);field(t,"phi",&e.phi,"D",w);
}
void time_fields(TTree *t,TimeRow &r,bool w) {
  field(t,"tai_sec",&r.tai_sec,"L",w);field(t,"tai_subsec",&r.tai_subsec,"D",w);
  field(t,"unix_sec",&r.unix_sec,"L",w);field(t,"unix_subsec",&r.unix_subsec,"D",w);
  field(t,"mjd",&r.mjd,"D",w);field(t,"mjdns4",&r.mjdns4,"I",w);
  field(t,"ra_mean",&r.ra_mean,"D",w);field(t,"dec_mean",&r.dec_mean,"D",w);
  field(t,"ra_mean_deg",&r.ra_mean_deg,"D",w);field(t,"dec_mean_deg",&r.dec_mean_deg,"D",w);
  field(t,"match_status",&r.match_status,"I",w);
}
std::unique_ptr<TFile> input_file(const std::string &path) {
  auto f=std::unique_ptr<TFile>(TFile::Open(path.c_str(),"READ"));
  require(f && !f->IsZombie() && !f->TestBit(TFile::kRecovered),"Unreadable/incomplete ROOT: "+path);
  require(f->GetEND()<=f->GetSize(),"ROOT end boundary exceeds file size: "+path);
  return f;
}
TTree *tree_in(TFile &file,const char *name) {
  auto t=dynamic_cast<TTree*>(file.Get(name)); require(t!=nullptr,"Missing tree: "+std::string(name));
  t->SetBranchStatus("*",0);return t;
}
void convert_time(TimeRow &r) {
  Ttai_t tai(r.tai_sec,r.tai_subsec); Tunix_t unix_time(tai); Tmjd_t mjd_time(tai);
  r.unix_sec=unix_time.sec;r.unix_subsec=unix_time.subsec;r.mjd=mjd_time.unpack();r.mjdns4=mjd_time.getns4();
}
bool same_time(const TimeRow &a,const TimeRow &b) {
  return a.tai_sec==b.tai_sec && a.unix_sec==b.unix_sec && a.mjdns4==b.mjdns4 &&
    same_double(a.tai_subsec,b.tai_subsec) && same_double(a.unix_subsec,b.unix_subsec) && same_double(a.mjd,b.mjd);
}
bool valid_time(const TimeRow &r) {
  if(r.tai_sec<=0 || r.unix_sec<=0 || !std::isfinite(r.tai_subsec) ||
     r.tai_subsec<0 || r.tai_subsec>=1 || !std::isfinite(r.unix_subsec) ||
     r.unix_subsec<0 || r.unix_subsec>=1 || !std::isfinite(r.mjd) || r.mjd<=0) return false;
  TimeRow expected=r;convert_time(expected);return same_time(r,expected);
}
bool valid_coords(const TimeRow &r) {
  return std::isfinite(r.ra_mean) && std::isfinite(r.dec_mean) &&
    r.ra_mean>=0 && r.ra_mean<papi::twopi && std::abs(r.dec_mean)<=papi::halfpi &&
    same_double(r.ra_mean_deg,r.ra_mean*papi::raddeg) && same_double(r.dec_mean_deg,r.dec_mean*papi::raddeg);
}
bool same_coords(const TimeRow &a,const TimeRow &b) {
  return same_double(a.ra_mean,b.ra_mean) && same_double(a.dec_mean,b.dec_mean) &&
    same_double(a.ra_mean_deg,b.ra_mean_deg) && same_double(a.dec_mean_deg,b.dec_mean_deg);
}
bool coordinates(const Event &e,TimeRow &r) {
  if(!std::isfinite(e.theta)||!std::isfinite(e.phi)||e.theta<0||e.theta>papi::pi) return false;
  double ra_apparent=0,dec_apparent=0;
  papi::hcs2eqa(r.mjd,0.0,e.theta,papi::azimxwcda(e.phi),ra_apparent,dec_apparent);
  papi::eqa2eqm(r.mjd,0.0,ra_apparent,dec_apparent,r.ra_mean,r.dec_mean);
  r.ra_mean=papi::angfix(r.ra_mean);
  r.ra_mean_deg=r.ra_mean*papi::raddeg;r.dec_mean_deg=r.dec_mean*papi::raddeg;
  return valid_coords(r);
}
struct Legacy {
  Event event;TimeRow time;bool conflict=false,coords_proven=false,coords_ambiguous=false;
  Long64_t conflict_sec=0;double conflict_subsec=0;
};
double wall_seconds() {return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Stats {
  Long64_t entries=0,reused_time=0,reused_coords=0,new_time=0,new_coords=0;
  Long64_t coordinate_calls=0,coordinate_checks=0,ies_fallback=0,skipped_legacy_files=0;
  Long64_t status[7]={};
  Long64_t reduced_groups=0,read_bytes=0,max_rss_kib=0;
  double cpu_seconds=0;
  std::map<std::string,double> seconds;
};
std::string quote_json(const std::string &s) {
  std::ostringstream out;out<<'"';
  for(unsigned char c:s) {
    if(c=='"'||c=='\\') out<<'\\'<<c;
    else if(c<32) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
    else out<<c;
  }
  out<<'"';return out.str();
}

void verify_output(const std::string &path,const std::vector<Event> *targets=nullptr,
                   const std::string &identity="") {
  auto f=input_file(path);auto t=tree_in(*f,"t_recovered_time");auto a=tree_in(*f,"t_alignment");
  require(t->GetEntries()==a->GetEntries(),"Output trees misaligned");
  if(targets) require(t->GetEntries()==static_cast<Long64_t>(targets->size()),"Output count changed");
  if(!identity.empty()) {
    auto id=dynamic_cast<TNamed*>(f->Get("recovery_request_id"));
    require(id && identity==id->GetTitle(),"Wrong output request identity");
  }
  TimeRow row;Event event;Long64_t index=-1;
  time_fields(t,row,false);event_fields(a,event,false);field(a,"entry_index",&index,"L",false);
  for(Long64_t i=0;i<t->GetEntries();++i) {
    require(t->GetEntry(i)>0 && a->GetEntry(i)>0,"Output read failed");
    require(index==i && row.match_status>=0 && row.match_status<=6,"Invalid alignment/status");
    if(targets) {
      const auto &e=targets->at(i);
      require(e.key()==event.key() && same_double(e.theta,event.theta) && same_double(e.phi,event.phi),"Output event sequence changed");
    }
    if(row.match_status==0||row.match_status==6) require(valid_time(row),"Invalid output time");
    if(row.match_status==0) require(valid_coords(row),"Invalid output coordinates");
  }
  require(!root_error,"ROOT reported an error during complete readback");
}

class Recovery {
  std::map<EventKey,Legacy> old;
  std::string reduced_root,reduced_manifest_path,loaded_day;
  double index_seconds=0;Long64_t index_read_bytes=0;bool index_reported=false;
  std::map<Key,std::string> manifest;
  std::map<std::pair<int,int>,std::string> run_seq;
  struct CachedReduced {ReducedContext data;int status=1;bool fallback=false;};
  std::map<Key,CachedReduced> reduced_cache;
  Long64_t skipped=0;
public:
  Recovery(const std::string &old_manifest,const std::string &reduced_path,const std::string &reduced_manifest=""):reduced_root(reduced_path),reduced_manifest_path(reduced_manifest) {
    double index_started=wall_seconds();
    if(old_manifest.empty()) return;
    std::ifstream list(old_manifest);require(bool(list),"Missing legacy manifest");
    std::string line;
    while(std::getline(list,line)) {
      if(line.empty()) continue;
      std::istringstream fields(line);std::string main_path,friend_path;int reuse=0,proof=0;
      require(bool(fields>>main_path>>friend_path>>reuse>>proof),"Malformed legacy manifest");
      if(!reuse) {++skipped;continue;}
      auto main=input_file(main_path);auto recovered=input_file(friend_path);
      auto m=tree_in(*main,"t_eventout");auto f=tree_in(*recovered,"t_recovered_time");
      if(m->GetEntries()!=f->GetEntries()) {++skipped;continue;}
      Event event;TimeRow row;event_fields(m,event,false);time_fields(f,row,false);
      for(Long64_t i=0;i<m->GetEntries();++i) {
        require(m->GetEntry(i)>0 && f->GetEntry(i)>0,"Legacy read failed");
        if(row.match_status!=0 || !valid_time(row)) continue;
        auto it=old.find(event.key());
        if(it==old.end()) old.emplace(event.key(),Legacy{event,row,false,bool(proof)&&valid_coords(row),false});
        else {
          auto &previous=it->second;
          if(!same_time(previous.time,row)) {previous.conflict=true;previous.conflict_sec=row.tai_sec;previous.conflict_subsec=row.tai_subsec;}
          if(!same_double(previous.event.theta,event.theta)||!same_double(previous.event.phi,event.phi)||!same_coords(previous.time,row)) {
            previous.coords_ambiguous=true;previous.coords_proven=false;
          }
        }
      }
      require(!root_error,"ROOT error in legacy index");
      index_read_bytes+=main->GetBytesRead()+recovered->GetBytesRead();
    }
    index_seconds=wall_seconds()-index_started;
  }
  Stats run(const std::string &input,const std::string &output,Long64_t expected,
            const std::string &identity,const std::string &commit) {
    root_error=false;
    double total_started=wall_seconds();std::clock_t cpu_started=std::clock();
    reduced_cache.clear();Long64_t cache_bytes=0,reduced_read_bytes=0;
    FileMeta meta;require(parse_eval_filename(input,meta),"Invalid input filename");
    require(!path_exists(output),"Refusing to overwrite output");
    auto in=input_file(input);auto source=tree_in(*in,"t_eventout");Event event;
    event_fields(source,event,false);
    auto pinc=source->GetLeaf("pincness");require(pinc!=nullptr,"Missing pincness");
    require(std::string(pinc->GetTypeName())=="Double_t"||std::string(pinc->GetTypeName())=="Float_t","Wrong pincness type");
    require(source->GetBranch("pincness")->GetEntries()==source->GetEntries(),"Incomplete pincness branch");
    source->SetBranchStatus("pincness",1);
    auto out=std::unique_ptr<TFile>(TFile::Open(output.c_str(),"RECREATE"));
    require(out && !out->IsZombie(),"Cannot create output");
    TimeRow row;Long64_t index=0;TTree times("t_recovered_time","Recovered event time and J2000 coordinates");
    TTree alignment("t_alignment","Exact pincness-selected observation sequence");
    time_fields(&times,row,true);event_fields(&alignment,event,true);field(&alignment,"entry_index",&index,"L",true);
    std::ofstream exceptions(output+".exceptions.jsonl");require(bool(exceptions),"Cannot write exceptions");
    exceptions<<std::setprecision(17);
    Stats stats;stats.skipped_legacy_files=skipped;std::vector<Event> targets;
    stats.seconds["legacy_index"]=index_reported?0:index_seconds;
    stats.read_bytes=index_reported?0:index_read_bytes;index_reported=true;
    targets.reserve(expected>=0?expected:1024);
    for(Long64_t i=0;i<source->GetEntries();++i) {
      double scalar_started=wall_seconds();
      require(source->GetEntry(i)>0,"Raw scalar read failed");
      stats.seconds["target_scalar_read"]+=wall_seconds()-scalar_started;
      if(!(pinc->GetValue()<1.1)) continue;
      row=TimeRow{};bool fallback=false;std::string reduced_source;
      double matching_started=wall_seconds();
      auto hit=old.find(event.key());
      Legacy *legacy=hit==old.end()?nullptr:&hit->second;
      if(legacy && legacy->conflict) {
        std::ostringstream error;error<<std::setprecision(17)<<"Conflicting legacy times at raw entry "<<i<<" key=("<<event.irun<<","<<event.ies<<","<<event.iseq<<","<<event.ievent<<") tai=("<<legacy->time.tai_sec<<","<<legacy->time.tai_subsec<<") versus ("<<legacy->conflict_sec<<","<<legacy->conflict_subsec<<")";
        throw std::runtime_error(error.str());
      }
      if(legacy) {row=legacy->time;++stats.reused_time;stats.seconds["key_matching"]+=wall_seconds()-matching_started;}
      else {
        stats.seconds["key_matching"]+=wall_seconds()-matching_started;
        double missing_started=wall_seconds();
        if(loaded_day!=meta.mmdd) {
          manifest=reduced_manifest_path.empty()?build_manifest(join_path(reduced_root,meta.mmdd)):read_manifest_file(reduced_manifest_path,meta);
          require(!manifest.empty(),"No readable reduced manifest for "+meta.mmdd);
          run_seq=build_unique_run_seq_manifest(manifest);loaded_day=meta.mmdd;
        }
        Key key{event.irun,event.ies,event.iseq};
        auto cached=reduced_cache.find(key);
        if(cached==reduced_cache.end()) {
          CachedReduced value;MissingReducedCache missing;bool previous_error=root_error;
          auto rc=load_reduced(key,manifest,run_seq,value.data,missing,value.fallback);
          if(!rc) {root_error=previous_error;value.status=1;}
          else if(!rc->tree) value.status=2;
          else value.status=0;
          if(value.data.file) {reduced_read_bytes+=value.data.file->GetBytesRead();value.data.file.reset();value.data.tree=nullptr;}
          cache_bytes+=value.data.event_ids.size()*4+value.data.secs.size()*8+value.data.subsecs.size()*8;
          require(cache_bytes<=512LL*1024*1024,"Hourly reduced cache exceeds 512 MiB");
          cached=reduced_cache.emplace(key,std::move(value)).first;
        }
        auto &cached_value=cached->second;auto &rc=cached_value.data;
        fallback=cached_value.fallback;reduced_source=rc.path;
        if(cached_value.status) row.match_status=cached_value.status;
        else if(event.ievent<0||event.ievent>=rc.entries) row.match_status=3;
        else if(rc.event_ids[event.ievent]!=event.ievent) row.match_status=4;
        else {
          row.tai_sec=rc.secs[event.ievent];row.tai_subsec=rc.subsecs[event.ievent];
          if(row.tai_sec>0&&std::isfinite(row.tai_subsec)&&row.tai_subsec>=0&&row.tai_subsec<1) convert_time(row);
          if(!valid_time(row)) {row=TimeRow{};row.match_status=5;}
          else {++stats.new_time;if(fallback) ++stats.ies_fallback;}
        }
        stats.seconds["missing_time"]+=wall_seconds()-missing_started;
      }
      double coordinate_started=wall_seconds();
      if(row.match_status==0) {
        bool angles=legacy && !legacy->coords_ambiguous && same_double(event.theta,legacy->event.theta) && same_double(event.phi,legacy->event.phi);
        if(angles && legacy->coords_proven) ++stats.reused_coords;
        else {
          TimeRow computed=row;++stats.coordinate_calls;
          if(!coordinates(event,computed)) {
            row.ra_mean=row.dec_mean=row.ra_mean_deg=row.dec_mean_deg=0;row.match_status=6;
          } else if(angles && valid_coords(legacy->time) && same_coords(computed,legacy->time)) {
            // Recalculation establishes provenance when historical run flags are unavailable.
            legacy->coords_proven=true;++stats.coordinate_checks;++stats.reused_coords;
          } else {
            row.ra_mean=computed.ra_mean;row.dec_mean=computed.dec_mean;
            row.ra_mean_deg=computed.ra_mean_deg;row.dec_mean_deg=computed.dec_mean_deg;++stats.new_coords;
          }
        }
      }
      stats.seconds["coordinates"]+=wall_seconds()-coordinate_started;
      double writing_started=wall_seconds();
      if(row.match_status || fallback) {
        static const char *reason[]={"ies_fallback","missing_reduced","missing_leaf","entry_out_of_range","event_mismatch","invalid_time","invalid_coordinates"};
        std::string explanation=reason[row.match_status];
        if(row.match_status==1&&!reduced_source.empty()) explanation=fallback?"unreadable_ies_fallback":"unreadable_exact_reduced";
        exceptions<<"{\"entry_index\":"<<index<<",\"irun\":"<<event.irun<<",\"ies\":"<<event.ies<<",\"iseq\":"<<event.iseq<<",\"ievent\":"<<event.ievent
          <<",\"status\":"<<row.match_status<<",\"reason\":"<<quote_json(explanation)<<",\"tai_sec\":"<<row.tai_sec
          <<",\"tai_subsec\":"<<row.tai_subsec<<",\"unix_sec\":"<<row.unix_sec<<",\"unix_subsec\":"<<row.unix_subsec<<",\"mjd\":"<<row.mjd
          <<",\"ies_fallback\":"<<(fallback&&valid_time(row)?"true":"false")<<",\"time_source\":"<<quote_json(legacy?"legacy":"reduced")<<",\"reduced_source\":"<<quote_json(reduced_source)<<"}\n";
      }
      require(times.Fill()>=0 && alignment.Fill()>=0,"ROOT Fill failed");
      targets.push_back(event);++stats.status[row.match_status];++index;
      stats.seconds["output_fill"]+=wall_seconds()-writing_started;
    }
    stats.entries=index;
    require(expected<0||expected==index,"Target count differs from observation receipt");
    require(stats.reused_time+stats.new_time==stats.status[0]+stats.status[6],"Time accounting mismatch");
    require(stats.reused_coords+stats.new_coords==stats.status[0],"Coordinate accounting mismatch");
    double write_started=wall_seconds();
    out->cd();
    require(times.Write()>0 && alignment.Write()>0,"ROOT tree write failed");
    TNamed id("recovery_request_id",identity.c_str()),code("recovery_code_commit",commit.c_str());
    require(id.Write()>0 && code.Write()>0,"ROOT identity write failed");
    require(!out->TestBit(TFile::kWriteError)&&!root_error,"ROOT write error");
    out->Close();require(!out->TestBit(TFile::kWriteError),"ROOT close error");
    exceptions.close();require(bool(exceptions),"Exception write failed");
    stats.seconds["output_close"]=wall_seconds()-write_started;
    double validation_started=wall_seconds();
    verify_output(output,&targets,identity);
    stats.seconds["complete_validation"]=wall_seconds()-validation_started;
    stats.seconds["total"]=wall_seconds()-total_started;
    stats.reduced_groups=reduced_cache.size();stats.read_bytes+=in->GetBytesRead()+reduced_read_bytes;
    stats.cpu_seconds=double(std::clock()-cpu_started)/CLOCKS_PER_SEC;
    rusage usage{};getrusage(RUSAGE_SELF,&usage);stats.max_rss_kib=usage.ru_maxrss;
    std::ofstream summary(output+".summary.json");
    summary<<"{\"entries\":"<<stats.entries<<",\"reused_time\":"<<stats.reused_time<<",\"reused_coords\":"<<stats.reused_coords
      <<",\"new_time\":"<<stats.new_time<<",\"new_coords\":"<<stats.new_coords<<",\"coordinate_calls\":"<<stats.coordinate_calls
      <<",\"coordinate_checks\":"<<stats.coordinate_checks<<",\"ies_fallback\":"<<stats.ies_fallback<<",\"skipped_legacy_files\":"<<skipped<<",\"status\":[";
    for(int i=0;i<7;++i) summary<<(i?",":"")<<stats.status[i];
    summary<<"],\"reduced_groups\":"<<stats.reduced_groups<<",\"read_bytes\":"<<stats.read_bytes<<",\"max_rss_kib\":"<<stats.max_rss_kib<<",\"cpu_seconds\":"<<stats.cpu_seconds<<",\"seconds\":{";
    bool comma=false;for(const auto &item:stats.seconds) {summary<<(comma?",":"")<<quote_json(item.first)<<":"<<item.second;comma=true;}
    summary<<"}}\n";summary.close();require(bool(summary),"Summary write failed");
    return stats;
  }
};

void self_test(const std::string &directory);
void compare_reference(const std::string &actual,const std::string &reference) {
  auto f=input_file(actual),r=input_file(reference);auto a=tree_in(*f,"t_recovered_time"),b=tree_in(*r,"t_recovered_time");
  require(a->GetEntries()==b->GetEntries(),"Reference entry count differs");
  TimeRow x,y;time_fields(a,x,false);time_fields(b,y,false);
  for(Long64_t i=0;i<a->GetEntries();++i) {
    require(a->GetEntry(i)>0&&b->GetEntry(i)>0,"Reference read failed");
    require(same_time(x,y),"Reference time differs at entry "+std::to_string(i));
    if(x.match_status==6) require(y.match_status==0&&!valid_coords(y),"Unexpected reference coordinate failure difference");
    else {
      require(x.match_status==y.match_status,"Reference status differs at entry "+std::to_string(i));
      require(same_coords(x,y),"Reference coordinates differ at entry "+std::to_string(i));
    }
  }
  require(!root_error,"ROOT error during reference comparison");
  std::cout<<"REFERENCE_EXACT_OK entries="<<a->GetEntries()<<"\n";
}
} // namespace

int main(int argc,char **argv) {
  SetErrorHandler(error_handler);
  try {
    std::map<std::string,std::string> opt;
    for(int i=1;i<argc;++i) {
      std::string key=argv[i];
      if(key=="--serve") {opt[key]="1";continue;}
      require(i+1<argc,"Missing option value");opt[key]=argv[++i];
    }
    if(opt.count("--compare")) {compare_reference(opt.at("--compare"),opt.at("--reference"));return 0;}
    if(opt.count("--self-test")) {self_test(opt.at("--self-test"));return 0;}
    if(opt.count("--verify")) {verify_output(opt.at("--verify"));return 0;}
    Recovery engine(opt["--legacy-manifest"],opt.at("--reduced-root"),opt["--reduced-manifest"]);
    if(opt.count("--serve")) {
      std::cout<<"READY\n"<<std::flush;std::string line;
      while(std::getline(std::cin,line)) {
        std::istringstream fields(line);std::string input,output,id,commit;Long64_t expected;
        require(bool(fields>>input>>output>>expected>>id>>commit),"Malformed worker request");
        engine.run(input,output,expected,id,commit);
        std::cout<<"RESULT "<<output<<"\n"<<std::flush;
      }
    } else engine.run(opt.at("--input"),opt.at("--output"),std::stoll(opt.at("--expected-entries")),opt.at("--request-id"),opt.at("--git-commit"));
    return 0;
  } catch(const std::exception &e) {std::cerr<<"RECOVERY_FAILED "<<e.what()<<"\n";return 1;}
}

namespace {
void fixture_events(const std::string &path,const std::vector<Event> &events,const std::vector<double> &pinc={}) {
  gSystem->mkdir(dirname_of(path).c_str(),true);TFile file(path.c_str(),"RECREATE");
  TTree tree("t_eventout","fixture");Event event;double p=0.5;event_fields(&tree,event,true);
  tree.Branch("pincness",&p,"pincness/D");
  for(size_t i=0;i<events.size();++i) {event=events[i];p=pinc.empty()?0.5:pinc.at(i);require(tree.Fill()>=0,"Fixture Fill failed");}
  require(tree.Write()>0,"Fixture Write failed");file.Close();
}
void fixture_friend(const std::string &path,const std::vector<TimeRow> &rows) {
  TFile file(path.c_str(),"RECREATE");TTree tree("t_recovered_time","fixture");TimeRow row;time_fields(&tree,row,true);
  for(const auto &r:rows) {row=r;require(tree.Fill()>=0,"Fixture friend Fill failed");}
  require(tree.Write()>0,"Fixture friend Write failed");file.Close();
}
std::vector<TimeRow> fixture_rows(const std::string &path) {
  auto file=input_file(path);auto t=tree_in(*file,"t_recovered_time");TimeRow row;time_fields(t,row,false);
  std::vector<TimeRow> rows;
  for(Long64_t i=0;i<t->GetEntries();++i) {require(t->GetEntry(i)>0,"Fixture read failed");rows.push_back(row);}
  return rows;
}
void self_test(const std::string &directory) {
  const auto path=[&](const std::string &s){return join_path(directory,s);};
  gSystem->mkdir(path("reduced/0101").c_str(),true);
  std::string reduced_file=path("reduced/0101/ES.101.WCDA_EVENT.test.es-7.20220101000000.8.reduced.root");
  {TFile f(reduced_file.c_str(),"RECREATE");TTree t("wcda","fixture");int ievent=0;Long64_t sec=1640995237;double subsec=.125;
    t.Branch("ievent",&ievent,"ievent/I");t.Branch("sec",&sec,"sec/L");t.Branch("subsec",&subsec,"subsec/D");
    for(int i=0;i<3;++i) {ievent=i;sec=1640995237+i;t.Fill();}t.Write();f.Close();}
  Event a{101,7,8,0,.4,1.},c{101,7,8,1,.5,1.2},d{101,7,8,2,.6,1.4};
  std::string input=path("Esg20220101_00.root");
  fixture_events(input,{c,a,d,a,d,a},{.5,.5,.5,.5,1.1,std::numeric_limits<double>::quiet_NaN()});
  Recovery baseline("",path("reduced"));baseline.run(input,path("reference.root"),4,"reference","test");
  auto reference=fixture_rows(path("reference.root"));
  fixture_events(path("old.root"),{a,c});TimeRow bad;bad.match_status=1;
  fixture_friend(path("old.time.root"),{reference[1],bad});
  {std::ofstream list(path("old.tsv"));list<<path("old.root")<<" "<<path("old.time.root")<<" 1 1\n";}
  Recovery hybrid(path("old.tsv"),path("reduced"));
  auto stats=hybrid.run(input,path("hybrid.root"),4,std::string(32,'c'),std::string(40,'b'));
  require(stats.reused_time==2&&stats.new_time==2&&stats.reused_coords==2&&stats.new_coords==2,"C/A/D/A reuse accounting failed");
  fixture_events(path("observed/0101/Esg20220101_00.root"),{c,a,d,a});
  auto rows=fixture_rows(path("hybrid.root"));
  for(size_t i=0;i<rows.size();++i) require(same_time(rows[i],reference[i])&&same_coords(rows[i],reference[i]),"Hybrid differs from all-new reference");
  TimeRow no_coordinates=reference[1];no_coordinates.ra_mean=no_coordinates.dec_mean=no_coordinates.ra_mean_deg=no_coordinates.dec_mean_deg=0;
  fixture_events(path("zero_old.root"),{a});fixture_friend(path("zero_old.time.root"),{no_coordinates});
  {std::ofstream list(path("zero_old.tsv"));list<<path("zero_old.root")<<" "<<path("zero_old.time.root")<<" 1 0\n";}
  fixture_events(path("Esg20220101_05.root"),{a});
  Recovery zero_old(path("zero_old.tsv"),path("reduced"));
  auto zero_stats=zero_old.run(path("Esg20220101_05.root"),path("zero_old_result.root"),1,"zero","test");
  require(zero_stats.reused_time==1&&zero_stats.new_coords==1&&same_coords(fixture_rows(path("zero_old_result.root"))[0],reference[1]),"Zero placeholder coordinates reused without evidence");
  {std::ofstream list(path("unproven.tsv"));list<<path("old.root")<<" "<<path("old.time.root")<<" 1 0\n";}
  fixture_events(path("Esg20220101_06.root"),{a,a});Recovery unproven(path("unproven.tsv"),path("reduced"));
  auto unproven_stats=unproven.run(path("Esg20220101_06.root"),path("unproven.root"),2,"unproven","test");
  require(unproven_stats.reused_coords==2&&unproven_stats.coordinate_checks==1&&unproven_stats.coordinate_calls==1,"Unproven coordinates were not checked once before reuse");
  fixture_events(path("cross.root"),{d});fixture_friend(path("cross.time.root"),{reference[2]});
  {std::ofstream list(path("old.tsv"),std::ios::app);list<<path("cross.root")<<" "<<path("cross.time.root")<<" 1 1\n";}
  Recovery cross(path("old.tsv"),path("reduced"));
  require(cross.run(input,path("cross_result.root"),4,"cross","test").reused_time==3,"Same-day cross-file lookup failed");
  Event moved=a;moved.phi+=.3;fixture_events(path("Esg20220101_01.root"),{moved});
  Recovery moved_engine(path("old.tsv"),path("reduced"));
  auto moved_stats=moved_engine.run(path("Esg20220101_01.root"),path("moved.root"),1,"moved","test");
  require(moved_stats.reused_time==1&&moved_stats.new_coords==1,"Changed angles reused coordinates");
  moved.theta=std::numeric_limits<double>::quiet_NaN();fixture_events(path("Esg20220101_02.root"),{moved});
  auto invalid=moved_engine.run(path("Esg20220101_02.root"),path("invalid.root"),1,"invalid","test");
  require(invalid.status[6]==1&&fixture_rows(path("invalid.root"))[0].tai_sec==reference[1].tai_sec,"Coordinate failure lost valid time");
  fixture_friend(path("short.time.root"),{reference[1]});
  {std::ofstream list(path("short.tsv"));list<<path("old.root")<<" "<<path("short.time.root")<<" 1 1\n";}
  Recovery mismatch(path("short.tsv"),path("reduced"));
  auto short_stats=mismatch.run(input,path("short_result.root"),4,"short","test");
  require(short_stats.reused_time==0&&short_stats.skipped_legacy_files==1,"Mismatched legacy entries accepted");
  {std::ofstream list(path("fallback.tsv"));list<<path("old.root")<<" "<<path("old.time.root")<<" 0 0\n";}
  Recovery tainted(path("fallback.tsv"),path("reduced"));
  require(tainted.run(input,path("tainted.root"),4,"tainted","test").reused_time==0,"Unauditable historical fallback reused");
  TimeRow conflict=reference[1];++conflict.tai_sec;convert_time(conflict);
  fixture_events(path("conflict.root"),{a});fixture_friend(path("conflict.time.root"),{conflict});
  {std::ofstream list(path("old.tsv"),std::ios::app);list<<path("conflict.root")<<" "<<path("conflict.time.root")<<" 1 1\n";}
  Recovery conflicting(path("old.tsv"),path("reduced"));bool stopped=false;
  try{conflicting.run(input,path("conflict_result.root"),4,"conflict","test");}catch(const std::exception&){stopped=true;}
  require(stopped,"Conflicting legacy time published");
  Event fallback_event=a;fallback_event.ies=9;fixture_events(path("Esg20220101_03.root"),{fallback_event});
  std::string corrupt_file=path("reduced/0101/ES.101.WCDA_EVENT.test.es-9.20220101000000.8.reduced.root");
  {std::ofstream corrupt(corrupt_file);corrupt<<"broken";}
  Recovery unreadable("",path("reduced"));
  auto unreadable_stats=unreadable.run(path("Esg20220101_03.root"),path("unreadable.root"),1,"unreadable","test");
  require(unreadable_stats.status[1]==1&&unreadable_stats.ies_fallback==0,"Unreadable exact ies used another file");
  require(std::remove(corrupt_file.c_str())==0,"Fixture cleanup failed");
  Recovery unique("",path("reduced"));
  require(unique.run(path("Esg20220101_03.root"),path("unique.root"),1,"unique","test").ies_fallback==1,"Unique absent-ies fallback failed");
  {std::ofstream list(path("ambiguous.tsv"));
    list<<"20220101 101 7 8 "<<reduced_file<<"\n20220101 101 7 8 "<<reduced_file<<".duplicate\n";}
  bool ambiguity_stopped=false;
  try {
    Recovery ambiguous("",path("reduced"),path("ambiguous.tsv"));
    ambiguous.run(path("Esg20220101_03.root"),path("ambiguous.root"),1,"ambiguous","test");
  } catch(const std::exception&) {ambiguity_stopped=true;}
  require(ambiguity_stopped,"Duplicate reduced candidates silently selected");
  bool count_stopped=false;
  try{baseline.run(input,path("wrong_count.root"),5,"wrong","test");}catch(const std::exception&){count_stopped=true;}
  require(count_stopped,"Wrong target count published");
  bool overwrite_stopped=false;
  try{baseline.run(input,path("reference.root"),4,"overwrite","test");}catch(const std::exception&){overwrite_stopped=true;}
  require(overwrite_stopped,"Existing output overwritten");
  std::cout<<"ROOT_REGRESSIONS_OK order duplicate strict-cut NaN cross-file conflict count-mismatch angle-change coordinate-failure fallback unreadable-exact overwrite\n";
}
} // namespace
