#include <TFile.h>
#include <TTree.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string label_from_path(const std::string &path) {
    const std::size_t slash = path.find_last_of('/');
    const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const std::string suffix = "_BKG_J2000.root";
    if (name.size() <= suffix.size() || name.substr(name.size() - suffix.size()) != suffix) {
        return name;
    }
    return name.substr(0, name.size() - suffix.size());
}

}  // namespace

void audit_pass5_map_livetime(const char *map_list_path, const char *output_csv_path) {
    std::ifstream map_list(map_list_path);
    if (!map_list) {
        throw std::runtime_error(std::string("Cannot open map list: ") + map_list_path);
    }
    std::ofstream output(output_csv_path);
    if (!output) {
        throw std::runtime_error(std::string("Cannot open output CSV: ") + output_csv_path);
    }
    output << "label,map_uri,pass5_ltime_seconds,mjd0,mjd1\n";
    output << std::setprecision(17);

    std::string path;
    std::size_t count = 0;
    double total_ltime = 0.0;
    while (std::getline(map_list, path)) {
        if (path.empty()) {
            continue;
        }
        TFile *file = TFile::Open(path.c_str(), "READ");
        if (file == nullptr || file->IsZombie()) {
            throw std::runtime_error(std::string("Cannot open ROOT map: ") + path);
        }
        TTree *header = dynamic_cast<TTree *>(file->Get("bkg_header"));
        if (header == nullptr || header->GetEntries() != 1) {
            file->Close();
            delete file;
            throw std::runtime_error(std::string("Missing one-row bkg_header: ") + path);
        }

        double ltime = 0.0;
        double mjd0 = 0.0;
        double mjd1 = 0.0;
        header->SetBranchAddress("Ltime", &ltime);
        header->SetBranchAddress("MJD0", &mjd0);
        header->SetBranchAddress("MJD1", &mjd1);
        header->GetEntry(0);
        output << label_from_path(path) << ',' << path << ',' << ltime << ',' << mjd0 << ',' << mjd1 << '\n';
        total_ltime += ltime;
        ++count;
        file->Close();
        delete file;
    }

    std::cout << "PASS5_LIVETIME_AUDIT maps=" << count
              << " total_seconds=" << std::setprecision(17) << total_ltime << std::endl;
}
