void scan_j2000_integrity() {
    std::ifstream listFile("/tmp/accepted_maps_uris.txt");
    std::string uri;
    int total = 0, bad = 0, good = 0;
    std::ofstream badOut("/tmp/bad_j2000_files.txt");
    while (std::getline(listFile, uri)) {
        if (uri.empty()) continue;
        total++;
        TFile *f = TFile::Open(uri.c_str(), "READ");
        if (!f || f->IsZombie()) {
            bad++;
            badOut << uri << "\tOPEN_FAILED" << std::endl;
            if (f) delete f;
            continue;
        }
        TList *keys = f->GetListOfKeys();
        int nkeys = keys ? keys->GetEntries() : 0;
        bool hasMjd = f->GetListOfKeys()->Contains("hMJD_all_fine");
        if (nkeys < 22 || !hasMjd) {
            bad++;
            badOut << uri << "\tNKEYS=" << nkeys << "\tHAS_MJD=" << hasMjd << std::endl;
        } else {
            good++;
        }
        f->Close();
        delete f;
        if (total % 100 == 0) {
            std::cout << "progress " << total << " good=" << good << " bad=" << bad << std::endl;
        }
    }
    std::cout << "TOTAL=" << total << " GOOD=" << good << " BAD=" << bad << std::endl;
}
