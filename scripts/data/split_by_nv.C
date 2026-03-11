#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <iostream>
#include <vector>
#include <chrono>

void printProgressBar(double progress, double elapsed_sec) {
    const int barWidth = 40;
    int pos = barWidth * progress;
    printf("\r");
    printf("🚀 进度 [");
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) printf("█");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %6.2f%% | 已用时 %.1fs", progress * 100.0, elapsed_sec);
    fflush(stdout);
}

int count_existing(const TString& dir, const char* prefix) {
    // 统计现有文件，作为起始编号，避免覆盖
    TString cmd; cmd.Form("ls -1 %s/%s_*.root 2>/dev/null | wc -l", dir.Data(), prefix);
    TString s = gSystem->GetFromPipe(cmd);
    return s.Atoi();
}

void split_by_nv() {
    // === 配置 ===
    TString indir  = "/home/server/mydisk/WCDA_simulation/"; // 输入
    TString outdir = "/home/server/mydisk/WCDA_split/";       // 输出
    gSystem->mkdir(outdir, kTRUE);

    const int maxEventsPerFile = 3800;

    // 三个 bin 的输出目录
    TString d1 = Form("%snv_60_150",   outdir.Data());
    TString d2 = Form("%snv_150_500",  outdir.Data());
    TString d3 = Form("%snv_500_3000", outdir.Data());
    gSystem->mkdir(d1, kTRUE); printf("🗂️ 创建输出目录: %s\n", d1.Data());
    gSystem->mkdir(d2, kTRUE); printf("🗂️ 创建输出目录: %s\n", d2.Data());
    gSystem->mkdir(d3, kTRUE); printf("🗂️ 创建输出目录: %s\n", d3.Data());

    // 为每个 bin 建立“全局文件计数器”（跨输入文件累加）
    int next_idx_60_150   = count_existing(d1, "nv_60_150");
    int next_idx_150_500  = count_existing(d2, "nv_150_500");
    int next_idx_500_3000 = count_existing(d3, "nv_500_3000");

    // === 找到所有输入 ROOT ===
    printf("🔍 正在扫描目录: %s\n", indir.Data());
    TString cmd; cmd.Form("ls -1 %s/*.root 2>/dev/null", indir.Data());
    TObjArray* files = gSystem->GetFromPipe(cmd).Tokenize("\n");
    const int total_files = files->GetEntriesFast();
    if (total_files == 0) { printf("⚠️ 未找到任何 ROOT 文件！\n"); return; }

    printf("📊 共找到 %d 个文件，开始处理...\n", total_files);
    auto t0 = std::chrono::steady_clock::now();

    int file_count = 0;

    for (int i = 0; i < total_files; ++i) {
        TString path = ((TObjString*)files->At(i))->GetString();
        if (path.Length() == 0) continue;
        ++file_count;

        // 进度条
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printProgressBar((double)file_count / total_files, elapsed);

        // 打开文件/取树
        std::unique_ptr<TFile> fin(TFile::Open(path));
        if (!fin || fin->IsZombie()) { printf("\n❌ 无法打开 %s\n", path.Data()); continue; }
        TTree* tin = (TTree*)fin->Get("t_eventout");
        if (!tin) { printf("\n⚠️ 文件缺少 t_eventout: %s\n", path.Data()); continue; }

        int nv; tin->SetBranchAddress("nv", &nv);

        // 分类索引
        std::vector<Long64_t> idx1, idx2, idx3;
        Long64_t n = tin->GetEntries();
        for (Long64_t j = 0; j < n; ++j) {
            tin->GetEntry(j);
            if      (nv >= 60  && nv < 150)  idx1.push_back(j);
            else if (nv >= 150 && nv < 500)  idx2.push_back(j);
            else if (nv >= 500 && nv < 3000) idx3.push_back(j);
        }

        // 写函数：使用“全局计数器”命名，避免覆盖
        auto save_bin = [&](std::vector<Long64_t>& idx, const TString& dir,
                            int low, int high, int& next_idx) {
            if (idx.empty()) return;
            int nFiles = (idx.size() + maxEventsPerFile - 1) / maxEventsPerFile;
            for (int k = 0; k < nFiles; ++k) {
                Long64_t start = k * maxEventsPerFile;
                Long64_t end   = std::min(start + (Long64_t)maxEventsPerFile, (Long64_t)idx.size());
                TString out; out.Form("%s/nv_%d_%d_%06d.root", dir.Data(), low, high, next_idx++);
                TFile fout(out, "RECREATE");
                if (!fout.IsOpen()) { printf("\n❌ 无法创建: %s\n", out.Data()); continue; }
                TTree* tout = tin->CloneTree(0);
                for (Long64_t r = start; r < end; ++r) {
                    tin->GetEntry(idx[r]);
                    tout->Fill();
                }
                tout->Write();
                // 可选：降低压缩以提速
                // fout.SetCompressionLevel(1);
                fout.Close();
                printf("\n✅ 写出 %s (%lld ~ %lld)", out.Data(), start, end - 1);
            }
        };

        save_bin(idx1, d1, 60, 150,   next_idx_60_150);
        save_bin(idx2, d2, 150, 500,  next_idx_150_500);
        save_bin(idx3, d3, 500, 3000, next_idx_500_3000);
    }

    printf("\n🎉 所有文件处理完成：%d/%d\n", file_count, total_files);
    double total_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("⏱️ 总耗时: %.1f s\n", total_elapsed);
}
