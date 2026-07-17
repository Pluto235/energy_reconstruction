#define wcdaevent_skymap_Cod_2_cxx
# include "src/wcdaevent_skymap_Cod_2.h"

# include <iostream>
# include <algorithm>
# include <cmath>
# include <fstream>
# include <iomanip>
# include <string>
# include <utility>
# include <vector>

# include <TH2.h>
# include <TStyle.h>
# include <TCanvas.h>

//# include "src/pinc_v2.h"
# include "src/DI_Config.h"
# include "src/DI_Bkg_Cod.h"

using namespace std;

namespace {

typedef pair<double, double> GtiInterval;
vector<GtiInterval> gti_intervals;

bool load_gti(const char *path)
{
    ifstream input(path);
    if (!input){
        cerr << "Error : cannot open GTI file " << path << endl;
        return false;
    }

    double start = 0;
    double stop = 0;
    while (input >> start >> stop){
        if (!isfinite(start) || !isfinite(stop) || stop < start){
            cerr << "Error : invalid GTI interval " << setprecision(15)
                 << start << " " << stop << " in " << path << endl;
            return false;
        }
        gti_intervals.push_back(GtiInterval(start, stop));
    }
    if (gti_intervals.empty()){
        cerr << "Error : no GTI intervals in " << path << endl;
        return false;
    }

    sort(gti_intervals.begin(), gti_intervals.end());
    double duration_seconds = 0;
    for (size_t index = 0; index < gti_intervals.size(); ++index){
        if (index > 0 && gti_intervals[index].first < gti_intervals[index - 1].second){
            cerr << "Error : overlapping GTI intervals in " << path << endl;
            return false;
        }
        duration_seconds += (gti_intervals[index].second - gti_intervals[index].first) * 86400.;
    }
    cout << "GTI intervals = " << gti_intervals.size()
         << ", duration = " << setprecision(15) << duration_seconds << " seconds" << endl;
    return true;
}

bool in_gti(double mjd)
{
    vector<GtiInterval>::const_iterator it = upper_bound(
        gti_intervals.begin(),
        gti_intervals.end(),
        mjd,
        [](double value, const GtiInterval &interval){ return value < interval.first; }
    );
    if (it == gti_intervals.begin()) return false;
    --it;
    return mjd >= it->first && mjd <= it->second;
}

}

void wcdaevent_skymap_Cod_2::Loop(char *fconf, char *outf1, char *outf2, TH1D *hmjd_all_fine, char *flist)
{
    if (fChain == 0) return;
    Long64_t nentries = fChain->GetEntriesFast();
    Long64_t nbytes = 0, nb = 0;

    DI_Config cf;
    cf.Readin(fconf, 0);

    char fname[400];
    TTree *tlist = new TTree();
    tlist->ReadFile(flist, "fname/C");
    tlist->SetBranchAddress("fname", fname);
    tlist->GetEntry(0);
    cout<<fname<<endl;
    TFile *ftemp = TFile::Open(fname);
    TTree *ttemp = (TTree *) ftemp->Get("wcdaevent");
    double mjd_temp = 0;
    ttemp->SetBranchAddress("mjd", &mjd_temp);
    ttemp->GetEntry(0);
    double mjd0 = mjd_temp;
    cout<<Form("%.7lf", mjd0)<<endl;
    ftemp->Close();

    tlist->GetEntry(tlist->GetEntries()-1);
    cout<<fname<<endl;
    TFile *ftemp1 = TFile::Open(fname);
    TTree *ttemp1 = (TTree *) ftemp1->Get("wcdaevent");
    ttemp1->SetBranchAddress("mjd", &mjd_temp);
    ttemp1->GetEntry(ttemp1->GetEntries()-1);
    double mjd1 = mjd_temp;
    ftemp1->Close();

    cout<<Form("%.7lf", mjd1)<<endl;

    double histogram_before = hmjd_all_fine->Integral();
    for (int ibin = 1; ibin <= hmjd_all_fine->GetNbinsX(); ++ibin){
        if (hmjd_all_fine->GetBinContent(ibin) <= 0) continue;
        double absolute_mjd = mjd0 + hmjd_all_fine->GetBinCenter(ibin);
        if (!in_gti(absolute_mjd)) hmjd_all_fine->SetBinContent(ibin, 0);
    }
    cout << "GTI histogram counts before = " << setprecision(15) << histogram_before
         << ", after = " << hmjd_all_fine->Integral() << endl;

    /*fChain->GetEntry(0);
    double mjd0 = mjd;
    fChain->GetEntry(nentries-1);
    double mjd1 = mjd;*/

    DI_Bkg_Cod *bkg = new DI_Bkg_Cod();
    bkg->SetGPpara(cf.nhit, cf.maxpinc, cf.mincpt);
    bkg->Init(cf.Harange, cf.Rarange, cf.Decrange, cf.Wbin, cf.MaskMapfile, cf.maxzen);
    bkg->SetFilterPara(mjd0, cf.maxGap, cf.minDuration, cf.filterMode);
    bkg->SetBkgPara(cf.minzen, cf.maxzen, cf.minAccCorr, cf.minBkgCorr);

    if (cf.filterMode){

        cout<<" ****** Filter out data with parameters : "<<endl;
        cout<<"    MaxGap = "<<cf.maxGap<<" seconds"<<endl;
        cout<<"    MinDuration = "<<cf.minDuration<<" seconds"<<endl;
        cout<<" ************************"<<endl;

        TH1D *hmjd_all = new TH1D("hmjd_all", "Counting rate vs. time", int(1.2*86400+0.5), -0.1, 1.1);
        // filter out bad data
        for (Long64_t jentry=0; jentry<nentries;jentry++) {
            Long64_t ientry = LoadTree(jentry);
            if (ientry < 0) break;
            if (jentry%(nentries/100)==0)
                cout<<" Event loop : "<<jentry/(nentries/100)<<" % ... "<<endl;
            nb = fChain->GetEntry(jentry);   nbytes += nb;
            // if (Cut(ientry) < 0) continue;
            if (!in_gti(mjd)) continue;
            if (fitstat!=0) continue;
            //if (poolused!=7) continue;

            hmjd_all->Fill(mjd-mjd0);
        }

        bool fliterflag = bkg->FilterData(hmjd_all);
        if (!fliterflag){
            cerr<<" Waring : Bad data! Total live time smaller than 2 hours"<<endl;

            TFile *fout = TFile::Open(outf1, "RECREATE");
            fout->cd();
            TH1D *hmjd_filter = (TH1D *) bkg->hMJD_filter->Clone("hMJD_filter");
            hmjd_all->Write();
            hmjd_filter->Write();
            fout->Close();

            return;
        }

    }

    cout<<" ****** Estimate background with parameters : "<<endl;
    cout<<"    MinAccCorr = "<<cf.minAccCorr<<endl;
    cout<<"    MinBkgCorr = "<<cf.minBkgCorr<<endl;
    cout<<" ************************"<<endl;

    // estimate acceptance
    for (Long64_t jentry=0; jentry<nentries;jentry++) {
        Long64_t ientry = LoadTree(jentry);
        if (ientry < 0) break;
        if (jentry%(nentries/100)==0)
            cout<<" Event loop : "<<jentry/(nentries/100)<<" % ... "<<endl;
        nb = fChain->GetEntry(jentry);   nbytes += nb;
        // if (Cut(ientry) < 0) continue;
        if (!in_gti(mjd)) continue;
        if (fitstat!=0) continue;
        if (poolused!=7) continue;

        int nq03t30 = nq03t30s0 + nq03t30s1 + nq03t30s2;
        if (nq03t30<10) continue;

        //theta = theta*papi::raddeg;
        phi   = phi*papi::raddeg;
        ra  = ra*papi::raddeg;
        dec = dec*papi::raddeg;
        float compactness = 0;
        if (cf.mincpt[0]>0){
            float CXPE = TMath::Max(TMath::Max(qmaxcxr45t30s0[0], qmaxcxr45t30s1[0]), qmaxcxr45t30s2[0]);
            compactness = nq03t30/(CXPE+0.01);
        }

        bkg->ProcessEvent(nq03t30, mjd, theta, phi, ra, dec, pincness, compactness, rmds, dcedge, f5w);

    }

    // Correct Acceptance and output Acceptance map
    //bool mapflag = bkg->CorrectAcceptance();
        // out Acceptance map
    TFile *fout = TFile::Open(outf1, "RECREATE");
    fout->cd();
    TH2D *hacc[cf.Nnhit];
    TH1F *hmjd[cf.Nnhit];
    for (int ii=0;ii<cf.Nnhit;ii++){

        if (cf.maxpinc[ii]>0){
            hacc[ii] = new TH2D(Form("hacc_%d", ii), Form("%d_%d_pinc%.2lf", cf.nhit[ii], cf.nhit[ii+1], cf.maxpinc[ii]), bkg->Map.Nhabin, bkg->Map.Harange[0], bkg->Map.Harange[1], bkg->Map.Ndecbin, bkg->Map.Decrange[0], bkg->Map.Decrange[1]);
            hmjd[ii] = new TH1F(Form("hmjd_%d", ii), Form("%d_%d_pinc%.2lf", cf.nhit[ii], cf.nhit[ii+1], cf.maxpinc[ii]), 103680, -0.1, 1.1);
        }

        if (cf.mincpt[ii]>0){
            hacc[ii] = new TH2D(Form("hacc_%d", ii), Form("%d_%d_cpt%.lf", cf.nhit[ii], cf.nhit[ii+1], cf.mincpt[ii]), bkg->Map.Nhabin, bkg->Map.Harange[0], bkg->Map.Harange[1], bkg->Map.Ndecbin, bkg->Map.Decrange[0], bkg->Map.Decrange[1]);
            hmjd[ii] = new TH1F(Form("hmjd_%d", ii), Form("%d_%d_cpt%.lf", cf.nhit[ii], cf.nhit[ii+1], cf.mincpt[ii]), 103680, -0.1, 1.1);
        }

        for (int jj=0;jj<bkg->Map.Nhabin;jj++)
            for (int kk=0;kk<bkg->Map.Ndecbin;kk++)
                hacc[ii]->SetBinContent(jj+1, kk+1, bkg->Nacc[ii][jj*bkg->Map.Ndecbin+kk]);

        for (int jj=0;jj<103680;jj++)
            hmjd[ii]->SetBinContent(jj+1, bkg->Rate[ii][jj]*1.);

        hacc[ii]->Write();
        hmjd[ii]->Write();

    }
    if (cf.filterMode){
        TH1D *hmjd_filter = (TH1D *) bkg->hMJD_filter->Clone("hMJD_filter");
        hmjd_filter->Write();
    }
    fout->Close();

    /*if (!mapflag){
        cerr<<"Error : Bad map! Acceptance correction factor of some sky cells smaller than "<<cf.minAccCorr<<" ."<<endl;
        return;
    }*/

    // estimate background
    /*bool bkgflag = bkg->ProcessHistBkg();
    if (!bkgflag){
        cerr<<"Error : Bad map! Background correction factor of some sky cells smaller than "<<cf.minBkgCorr<<" ."<<endl;
        return;
    }*/
        // out
    TFile *fout2 = TFile::Open(outf2, "RECREATE");
    fout2->cd();
        // tree of header 
    char *fConfig = fconf;
    double ltime = bkg->EffLtime;
    TTree *bkg_header = new TTree("bkg_header", "bkg_header");
    bkg_header->Branch("fConfig", fConfig, "fConfig/C");
    bkg_header->Branch("MJD0", &mjd0, "MJD0/D");
    bkg_header->Branch("MJD1", &mjd1, "MJD1/D");
    bkg_header->Branch("Ltime", &ltime, "Ltime/D");
    bkg_header->Fill();
    bkg_header->Write();
        // Fill map
    TH2D *hon[cf.Nnhit];
    TH1F *hmjd_mask[cf.Nnhit];
    //TH2D *hbkg[cf.Nnhit];
    //TH2D *hoff[cf.Nnhit];
    for (int ii=0;ii<cf.Nnhit;ii++){

        if (cf.maxpinc[ii]>0){
            hon[ii] = new TH2D(Form("hon_%d", ii), Form("%d_%d_pinc%.2lf", cf.nhit[ii], cf.nhit[ii+1], cf.maxpinc[ii]), bkg->Map.Nrabin, bkg->Map.Rarange[0], bkg->Map.Rarange[1], bkg->Map.Ndecbin, bkg->Map.Decrange[0], bkg->Map.Decrange[1]);
            hmjd_mask[ii] = new TH1F(Form("hmjd_mask_%d", ii), Form("%d_%d_pinc%.2lf", cf.nhit[ii], cf.nhit[ii+1], cf.maxpinc[ii]), 1036800, -0.1, 1.1);
            //hbkg[ii] = new TH2D(Form("hbkg_%d", ii), Form("%d_%d_pinc%.2lf", cf.nhit[ii], cf.nhit[ii+1], cf.maxpinc[ii]), bkg->Map.Nrabin, bkg->Map.Rarange[0], bkg->Map.Rarange[1], bkg->Map.Ndecbin, bkg->Map.Decrange[0], bkg->Map.Decrange[1]);
            //hoff[ii] = new TH2D(Form("hoff_%d", ii), Form("%d_%d_pinc%.2lf", cf.nhit[ii], cf.nhit[ii+1], cf.maxpinc[ii]), bkg->Map.Nrabin, bkg->Map.Rarange[0], bkg->Map.Rarange[1], bkg->Map.Ndecbin, bkg->Map.Decrange[0], bkg->Map.Decrange[1]);
        }

        if (cf.mincpt[ii]>0){
            hon[ii] = new TH2D(Form("hon_%d", ii), Form("%d_%d_cpt%.lf", cf.nhit[ii], cf.nhit[ii+1], cf.mincpt[ii]), bkg->Map.Nrabin, bkg->Map.Rarange[0], bkg->Map.Rarange[1], bkg->Map.Ndecbin, bkg->Map.Decrange[0], bkg->Map.Decrange[1]);
            //hbkg[ii] = new TH2D(Form("hbkg_%d", ii), Form("%d_%d_cpt%.lf", cf.nhit[ii], cf.nhit[ii+1], cf.mincpt[ii]), bkg->Map.Nrabin, bkg->Map.Rarange[0], bkg->Map.Rarange[1], bkg->Map.Ndecbin, bkg->Map.Decrange[0], bkg->Map.Decrange[1]);
            //hoff[ii] = new TH2D(Form("hoff_%d", ii), Form("%d_%d_cpt%.lf", cf.nhit[ii], cf.nhit[ii+1], cf.mincpt[ii]), bkg->Map.Nrabin, bkg->Map.Rarange[0], bkg->Map.Rarange[1], bkg->Map.Ndecbin, bkg->Map.Decrange[0], bkg->Map.Decrange[1]);
        }

        for (int jj=0;jj<bkg->Map.Nrabin;jj++)
            for (int kk=0;kk<bkg->Map.Ndecbin;kk++){
                hon[ii]->SetBinContent(jj+1, kk+1, bkg->Non[ii][jj*bkg->Map.Ndecbin+kk]*1.);
                //hbkg[ii]->SetBinContent(jj+1, kk+1, bkg->Nbkg[ii][jj*bkg->Map.Ndecbin+kk]);
                //hoff[ii]->SetBinContent(jj+1, kk+1, bkg->Noff[ii][jj*bkg->Map.Ndecbin+kk]);
            }
        
        for (int jj=0;jj<1036800;jj++)
            hmjd_mask[ii]->SetBinContent(jj+1, bkg->Rate_masked[ii][jj]*1.);

        hon[ii]->Write();
        hmjd_mask[ii]->Write();
        //hbkg[ii]->Write();
        //hoff[ii]->Write();

    }

    TH1D *hMJD_temp = (TH1D *) hmjd_all_fine->Clone("hMJD_all_fine");
    hMJD_temp->Write();

    fout2->Close();

}

bool inputcheck(char *fname){
    TFile *fin = TFile::Open(fname);
    if (!fin){
        cerr<<"Error : "<<fname<<" cannot be opened. Continued!"<<endl;
        return 0;}   
    if (fin->IsZombie()){
        cerr<<"Error : "<<fname<<" is Zombie. Continued!"<<endl;
        fin->Close();
        return 0;}   
    if (fin->GetEND()<10000){
        cerr<<"Error : "<<fname<<" is small. Continued!"<<endl;
        fin->Close();
        return 0;}   

    TObject *tob=(TObject*)fin->Get("wcdaevent");
    if (tob == 0) {
        cerr<<"Error : "<<fname<<" has no tree wcdaevent. Continued!"<<endl;
        fin->Close();
        return 0;}  

    fin->Close();

    return 1;
}

int main(int argc, char * argv[])
{ 
    if (argc!=7) {
        cerr<<" *** main : Error : too few input parameters"<<endl;
        cerr<<" *** main : "<<argv[0]<<"\n  [ inputfilelist : xxx.list ]\n  [ configfile : xxx.txt]\n  [ out1 : xxx.root (acceptance map) ]\n  [ out2 : xxx.root (sky map: on, bk, off) ]\n  [ inputDatalist : xxx.list ]\n  [ GTI file : start_mjd stop_mjd per line ]"<<endl;
        return -1; 
    }

    if (!load_gti(argv[6])) return -1;

    // input configfile
    DI_Config cf;
    bool cfflag = cf.Readin(argv[2], 1);
    if (!cfflag){
        cerr<<" Error : bad config!"<<endl;
        return -1;
    }

    // input : data
    // background estimation - based on Direct integral (DI) method
    char fname[200]; int ifile = 0;
    ifstream flist(argv[1]);
    TChain *finlist = new TChain();
    TTree  *tinlist = 0;
    TH1D *hmjd_all_fine = new TH1D("hmjd_all_fine", "Counting rate vs. time", int(1.2*864000+0.5), -0.1, 1.1);
    while (flist.getline(fname, 200)){
        if (inputcheck(fname)){
            cout<<" file "<<ifile++<<" : "<<fname<<endl;
            finlist->AddFile(fname, 0, "wcdaevent");

            TFile *fin = TFile::Open(fname);
            TH1D *hmjd_temp = (TH1D *) fin->Get("hMJD_all_fine");
            hmjd_all_fine->Add(hmjd_temp, 1);
            fin->Close();
        }
    }

    std::cout<<" *** main : Loop begins ... "<<std::endl;
    tinlist = (TTree *)finlist;
    std::cout<<" *** main : entries = "<<tinlist->GetEntries()<<std::endl;
    wcdaevent_skymap_Cod_2 fillmap(tinlist);
    fillmap.Loop(argv[2], argv[3], argv[4], hmjd_all_fine, argv[5]);

    cout<<"over"<<endl;
    return 0;     
}
