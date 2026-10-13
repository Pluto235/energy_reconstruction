#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <algorithm>

#include "TFile.h"
#include "TObject.h"
#include "TString.h"
#include "TSystem.h"
#include "TTree.h"
#include "TTreeFormula.h"
#include "TTreeCache.h"
#include "TLeaf.h"

struct ObservationFilterStats {
  Long64_t input_entries = 0, selected_entries = 0, header_entries = 0, input_bytes = 0;
  Long64_t scan_bytes = 0, hit_bytes = 0, scan_calls = 0, hit_calls = 0;
  double scan_seconds = 0, hit_seconds = 0;
  int scan_cached_branches = 0, hit_cached_branches = 0, write_errno = 0;
  std::vector<std::string> header_branches;
};

ObservationFilterStats &observation_filter_stats() {
  static ObservationFilterStats stats;
  return stats;
}

namespace {

std::vector<std::string> kept_event_branches_loop_scoped() {
  return {
      "n",          "nfit0",       "nfit",        "nfitb",      "vnfit",
      "fitstat",    "nrange",      "nv",          "vidmc",      "vx",
      "vy",         "vt",          "vq",          "theta",      "phi",
      "xc",         "yc",          "dcedge",      "dcedgepool", "istationcore",
      "ccindex",    "chi2",        "rmds",        "pincness",   "compactness",
      "f5w",        "irun",        "iseq",        "ies",        "ievent",
  };
}

void ensure_parent_dir_loop_scoped(const TString &output_path) {
  TString output_dir = gSystem->DirName(output_path);
  if (!output_dir.IsNull() && output_dir != ".") {
    gSystem->mkdir(output_dir, kTRUE);
  }
}

}  // namespace

int filter_observed_pincness_loop_scoped(const char *input_path,
                                         const char *output_path,
                                         double pincness_max = 1.1,
                                         int hit_cache_mb = -1) {
  auto &stats = observation_filter_stats();
  stats = ObservationFilterStats{};
  if (!input_path || !output_path) {
    std::cerr << "input_path and output_path are required" << std::endl;
    return 1;
  }

  TFile *input_file = TFile::Open(input_path, "READ");
  if (!input_file || input_file->IsZombie() || input_file->TestBit(TFile::kRecovered)) {
    std::cerr << "failed to open input ROOT file: " << input_path << std::endl;
    return 2;
  }

  TTree *event_tree = dynamic_cast<TTree *>(input_file->Get("t_eventout"));
  if (!event_tree) {
    std::cerr << "missing t_eventout in input ROOT file: " << input_path << std::endl;
    input_file->Close();
    return 3;
  }

  TTree *header_tree = dynamic_cast<TTree *>(input_file->Get("t_eventout_h"));
  if (!header_tree) {
    std::cerr << "missing t_eventout_h in input ROOT file: " << input_path << std::endl;
    input_file->Close();
    return 4;
  }

  const Long64_t input_entries = event_tree->GetEntries();
  stats.input_entries = input_entries;
  stats.input_bytes = input_file->GetSize();
  stats.header_entries = header_tree->GetEntries();
  for (auto *branch : *header_tree->GetListOfBranches()) stats.header_branches.push_back(branch->GetName());
  for (const auto &name : kept_event_branches_loop_scoped()) {
    if (!event_tree->GetBranch(name.c_str())) { input_file->Close(); return 7; }
  }
  const TString selection = Form("pincness < %.17g", pincness_max);

  event_tree->SetBranchStatus("*", 0);
  if (event_tree->GetBranch("pincness")) {
    event_tree->SetBranchStatus("pincness", 1);
  } else {
    std::cerr << "missing pincness branch" << std::endl;
    input_file->Close();
    return 7;
  }

  std::vector<Long64_t> selected_entries;
  selected_entries.reserve(input_entries / 20 + 1024);
  if (hit_cache_mb >= 0) {
    event_tree->SetCacheSize(0);
    event_tree->SetCacheSize(8 * 1024 * 1024);
    event_tree->AddBranchToCache("pincness", true);
    event_tree->StopCacheLearningPhase();
  }
  const auto scan_start = std::chrono::steady_clock::now();
  const auto scan_bytes_start = input_file->GetBytesRead();
  const auto scan_calls_start = input_file->GetReadCalls();
  {
    TTreeFormula select_formula("select_formula", selection, event_tree);
    for (Long64_t i = 0; i < input_entries; ++i) {
      if (event_tree->GetEntry(i) <= 0) {
        input_file->Close();
        return 8;
      }
      if (select_formula.EvalInstance() != 0.0) {
        selected_entries.push_back(i);
      }
      if ((i + 1) % 1000000 == 0) {
        std::cout << "SCAN_PROGRESS " << (i + 1) << "/" << input_entries
                  << " selected=" << selected_entries.size() << std::endl;
      }
    }
  }

  stats.selected_entries = selected_entries.size();
  stats.scan_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - scan_start).count();
  stats.scan_bytes = input_file->GetBytesRead() - scan_bytes_start;
  stats.scan_calls = input_file->GetReadCalls() - scan_calls_start;
  if (auto *cache = dynamic_cast<TTreeCache *>(input_file->GetCacheRead(event_tree)))
    stats.scan_cached_branches = cache->GetCachedBranches()->GetEntries();

  event_tree->SetBranchStatus("*", 0);
  const std::vector<std::string> keep_branches = kept_event_branches_loop_scoped();
  for (size_t i = 0; i < keep_branches.size(); ++i) {
    if (event_tree->GetBranch(keep_branches[i].c_str())) {
      event_tree->SetBranchStatus(keep_branches[i].c_str(), 1);
    }
  }
  if (hit_cache_mb >= 0) {
    event_tree->SetCacheSize(0);
    if (hit_cache_mb > 0) {
      event_tree->SetCacheSize(Long64_t(hit_cache_mb) * 1024 * 1024);
      event_tree->SetCacheEntryRange(0, input_entries);
      for (const auto &name : keep_branches) event_tree->AddBranchToCache(name.c_str(), true);
      event_tree->StopCacheLearningPhase();
    }
  }
  if (auto *cache = dynamic_cast<TTreeCache *>(input_file->GetCacheRead(event_tree)))
    stats.hit_cached_branches = cache->GetCachedBranches()->GetEntries();
  const auto hit_start = std::chrono::steady_clock::now();
  const auto hit_bytes_start = input_file->GetBytesRead();
  const auto hit_calls_start = input_file->GetReadCalls();

  ensure_parent_dir_loop_scoped(output_path);
  TFile *output_file = TFile::Open(output_path, "RECREATE");
  if (!output_file || output_file->IsZombie()) {
    std::cerr << "failed to create output ROOT file: " << output_path << std::endl;
    input_file->Close();
    return 5;
  }

  output_file->cd();
  TTree *filtered_tree = event_tree->CloneTree(0);
  if (!filtered_tree) {
    std::cerr << "CloneTree(0) failed" << std::endl;
    output_file->Close();
    input_file->Close();
    return 6;
  }
  filtered_tree->SetName("t_eventout");
  filtered_tree->SetDirectory(output_file);

  for (size_t j = 0; j < selected_entries.size(); ++j) {
    if (event_tree->GetEntry(selected_entries[j]) <= 0 ||
        filtered_tree->Fill() < 0 || output_file->TestBit(TFile::kWriteError)) {
      stats.write_errno = errno;
      std::cerr << "event read/write failed at selected entry " << j << std::endl;
      output_file->Close();
      input_file->Close();
      return 9;
    }
    if ((j + 1) % 50000 == 0) {
      std::cout << "FILL_PROGRESS " << (j + 1) << "/" << selected_entries.size()
                << std::endl;
    }
  }

  TTree *copied_header = header_tree->CloneTree(-1, "fast");
  if (!copied_header) {
    output_file->Close();
    input_file->Close();
    return 10;
  }
  copied_header->SetName("t_eventout_h");
  copied_header->SetDirectory(output_file);

  const auto written = output_file->Write("", TObject::kOverwrite);
  if (written <= 0 || output_file->TestBit(TFile::kWriteError)) stats.write_errno = errno;
  output_file->Close();
  const bool write_failed = written <= 0 || output_file->TestBit(TFile::kWriteError);
  if (write_failed && !stats.write_errno) stats.write_errno = errno;
  stats.hit_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - hit_start).count();
  stats.hit_bytes = input_file->GetBytesRead() - hit_bytes_start;
  stats.hit_calls = input_file->GetReadCalls() - hit_calls_start;
  input_file->Close();
  if (write_failed) return 11;

  std::cout << "INPUT " << input_path << std::endl;
  std::cout << "OUTPUT " << output_path << std::endl;
  std::cout << "ENTRIES_BEFORE " << input_entries << std::endl;
  std::cout << "ENTRIES_AFTER " << selected_entries.size() << std::endl;
  std::cout << "CUT " << selection << std::endl;
  std::cout << "DROPPED_BRANCHES vflag,vnpe,vqsamp" << std::endl;
  return 0;
}

bool readable_observation_tree(TTree *tree) {
  tree->SetBranchStatus("*", 1);
  for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
    if (tree->GetEntry(i) <= 0) return false;
  }
  return true;
}

bool checked_observation_content(TTree *tree, bool full, bool check_pincness) {
  tree->SetBranchStatus("*", 1);
  const Long64_t n = tree->GetEntries(), step = full ? 1 : std::max(Long64_t(1), n / 2048);
  auto *pincness = tree->GetLeaf("pincness");
  for (Long64_t i = 0; i < n; i += step) {
    if (tree->GetEntry(i) <= 0 || (check_pincness && (!pincness || !(pincness->GetValue() < 1.1)))) return false;
  }
  if (n && (tree->GetEntry(n - 1) <= 0 || (check_pincness && !(pincness->GetValue() < 1.1)))) return false;
  return true;
}

bool same_observation_tree(TTree *left, TTree *right) {
  if (left->GetEntries() != right->GetEntries() || left->GetListOfBranches()->GetEntries() != right->GetListOfBranches()->GetEntries()) return false;
  for (Long64_t i = 0; i < left->GetEntries(); ++i) {
    if (left->GetEntry(i) <= 0 || right->GetEntry(i) <= 0) return false;
    for (auto *item : *left->GetListOfLeaves()) {
      auto *a = static_cast<TLeaf *>(item), *b = right->GetLeaf(a->GetName());
      if (!b || std::string(a->GetTypeName()) != b->GetTypeName() || a->GetNdata() != b->GetNdata()) return false;
      for (int j = 0; j < a->GetNdata(); ++j) {
        const auto x = a->GetValue(j), y = b->GetValue(j);
        if (x != y && !(std::isnan(x) && std::isnan(y))) return false;
      }
    }
  }
  return true;
}
