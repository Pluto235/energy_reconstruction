#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TObject.h"
#include "TString.h"
#include "TSystem.h"
#include "TTree.h"
#include "TTreeFormula.h"

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
                                         double pincness_max = 1.1) {
  if (!input_path || !output_path) {
    std::cerr << "input_path and output_path are required" << std::endl;
    return 1;
  }

  TFile *input_file = TFile::Open(input_path, "READ");
  if (!input_file || input_file->IsZombie()) {
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

  event_tree->SetBranchStatus("*", 0);
  const std::vector<std::string> keep_branches = kept_event_branches_loop_scoped();
  for (size_t i = 0; i < keep_branches.size(); ++i) {
    if (event_tree->GetBranch(keep_branches[i].c_str())) {
      event_tree->SetBranchStatus(keep_branches[i].c_str(), 1);
    }
  }

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
  output_file->Close();
  const bool write_failed = written <= 0 || output_file->TestBit(TFile::kWriteError);
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
