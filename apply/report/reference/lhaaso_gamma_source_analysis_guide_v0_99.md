[toc]

# LHAASO gamma源分析程序使用指南 (v0.99)

`对本文档内容有任何疑问，请联系E-mail hushicong@ihep.ac.cn`
[English verison](https://jupyter.ihep.ac.cn/s/I55_G9oED)

原理参考[原理文档](https://jupyter.ihep.ac.cn/s/RFFxPeYYY)第一部分和第三部分，本文档主要介绍基于此原理开发的分析软件的使用方法。
*    软件环境（建议）：
`source /cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/envf.sh`
*    软件基于C++（及分析软件[ROOT](https://root.cern/)）实现，ROOT入门可参考[ROOT用户手册](https://root.cern/manual/)
*    该版本的配置文件采用yaml格式，简要说明可参考[yaml-cpp](https://cloud.tencent.com/developer/article/1423468)
*    高能所计算集群使用指南（提交作业、查看作业进度等等）：http://afsapply.ihep.ac.cn/cchelp/zh/experiments/LHAASO/
        *    <font color=red>本地运行被killed</font>：ROI较大或ROI内源的数量较多时，可能会遇到本地运行被killed的情况（可能是因为时间过长或内存占用过大），可提交作业运行。提交作业的命令如下：
                ```bash=
                hep_sub -g lhaaso -mem 15000 Src_Main -argu Fit.yaml ParInit.yaml
                    -g lhaaso  : 指定用户组
                    -mem 15000 : 指定最大运行内存（单位兆比特），可根据ROI或源的数量调整。15000对于大部分情况是足够的。大内存作业一般排队时间较长……
                ```
*    <font color=red>强烈建议</font>：先用这套程序做出Crab能谱（包括能谱点），确认无误后再分析其他源
*    分析复杂的区域之前，可以先看看本文档最后的使用建议

## <font color=Orange>更新</font>
*    一键更新：将脚本`Update.sh`拷贝到自己的目录下并运行后，source软件环境即可开始分析
*    [更新历史](https://jupyter.ihep.ac.cn/s/4_80LHQ3J)
*    更新日期：2025/08/10
*    更新内容
        *    版本：<font color=red>v0.9 -> v0.99</font>
                *    兼容v0.9版本的配置文件
        *    数据：
                *    WCDA：增加了新版本（Cod版本）到20240731的数据。
                *    KM2A：增加了20240731的数据
        *    程序/功能：   
                *    <font color=red>速度提升</font>：基于TMinuit的迭代策略，大幅加速了似然拟合的迭代速度，复杂多源区域的分析<font color=red>~10 times faster</font>
                *    更正了内存管理上的一些bug
                *    Parameters Link：在ParInit.yaml增加了link parameters的功能，可以用来测试可能统一起源的嵌套源或临近源的能谱的一致性，也为后续stacking analysis做准备
                *    ParInit.yaml中整合了LHAASO源表，可以直接从LHAASO源表结果出发做分析
                *    整理了程序的输出到输出路径下的两个文件夹中
                        *    OutDir/Check：显著性图和残差显著性二维天图和一维分布，用作初步的拟合结果的检查
                        *    OutDir/SED_Mor：自动生成的能谱图和存在txt文件中的能谱点数据
        *    使用文档：
                *    根据数据和程序的更新做了更新

## <font color=Orange>简介</font>
### <font color=violet>程序框架</font>
<img src="https://jupyter.ihep.ac.cn/uploads/4bd192cf-dc4d-4536-94c0-d6d1e8934ed2.png" width=90%>

*    详细软件架构见：[LA-Gamma](https://jupyter.ihep.ac.cn/s/I2NAO7b_d)

### <font color=violet>核心功能</font>
联合WCDA+KM2A，同时拟合单个或多个源或弥散（源或弥散均当作一个成分），给出以下结果：
*    参数
        *    每个成分的能谱参数和空间参数最优值及误差
*    TS值
        *    每个成分的TS值
        *    每个成分在每个能段的TS
*    能谱点
        *    每个成分在每个能段的中值能量和流强点/流强上限
*    形态
        *    每个源三个能段的显著性天图。三个能段为WCDA一个（nhit>=200）、KM2A两个（25TeV-100TeV、>=100TeV）
        *    每个能段数据的显著性
        *    每个源的TS map
        *    每个源的流强的径向分布
*    其他：
        *    参数CORRELATION Contour （较为耗时，暂时没加上）

### <font color=violet>支持的模型</font>
*    目前可选的空间模型如下（具体可见程序目录下的配置文件`src/Src_MorModel.yaml`）。<font color=Red>空间模型所有参数的单位均为度</font>
        | 模型描述   | 程序中的tag |说明|
        | :--------: | :--------: | :--------: |
        | 点源     | Point     ||
        | 对称高斯（解析卷积）| Ext_gaus||
        | 能量依赖的扩展     | Ext_gaus_E | 这里的能量依赖是指WCDA和KM2A扩展度不同，正在添加|
        | 对称高斯（数值卷积）| Ext_Gaus||
        | 椭高斯| Ext_EGaus| 参数Theta为长轴与ra正向的夹角
        | 均匀盘  | Ext_Disk||
        | 扩散模型|Ext_Diff|  取自arXiv:2106.09396 |
        | 能量依赖的$\theta_{d}$ | Ext_Diff_E | 这里的能量依赖是指WCDA和KM2A能段的$\theta_{d}$不同，正在添加 | 
        | 任意固定模板|Ext_Temp| 类似弥散伽马的拟合，可以用一个固定模板拟合源的形态 |
        | 均匀环|Ext_Ring| new |
        | 壳模型|Ext_Shell| new |
        | 均匀扇形|Ext_HDisk| new |
*    目前可选的能谱模型如下（具体可见程序目录下的配置文件`src/Src_SEDModel.yaml`）
        | 模型描述   | 程序中的tag |公式|
        | :--------: | :--------: | :--------: |
        |幂律谱|PL|$$F=F_{0}\left(\frac{E}{E_{\rm{piv}}}\right)^{-\alpha}$$|
        |对数抛物线|LP|$$F=F_{0}\left(\frac{E}{E_{\textrm{piv}}}\right)^{-\alpha-\beta \rm{log}\frac{E}{E_{\rm{piv}}}}$$|
        |带指数截断的幂律谱|PEC|$$F=F_{0}\left(\frac{E}{E_{\rm{piv}}}\right)^{-\alpha}\rm{exp}^{-\frac{E}{E_{\rm{cut}}}}$$|
        |慢指数截断的幂律谱|PSEC|$$F=F_{0}\left(\frac{E}{E_{\rm{piv}}}\right)^{-\alpha}\rm{exp}^{-\sqrt{\frac{E}{E_{\rm{cut}}}}}$$|
        |尖锐拐折的双幂律谱|BPL|$$F = F_{0}\left(\frac{E}{E_{\rm{b}}}\right)^{-\alpha_{1}} (E\le E_{\rm{b}}) \\ = \left(\frac{E}{E_{\rm{b}}}\right)^{-\alpha_{2}} (E\gt E_{\rm{b}})$$ |
        |平滑拐折的双幂律谱|SBPL|$$F=F_{0}\left(\frac{E}{E_{\rm{b}}}\right)^{-\alpha_{1}}\Bigg\{\frac{1}{2}\bigg[1+\left(\frac{E}{E_\rm{b}}\right)^{\frac{1}{\beta}}\bigg]\Bigg\}^{(\alpha_{1}-\alpha_{2})\beta}$$|

*    <font color=red>用户可按配置文件中的格式添加自己的模型</font>
        *    空间模型配置文件中的'NDim'对应模型公式中的自变量个数：例如对称高斯只有一个自变量，角距离；而椭高斯有两个自变量，角距离和垂直于视线方向的坐标平面上的方位角
        *    能谱模型配置文件中'Epiv'即pivot energy
        *    无需修改程序的其他部分，在模型配置文件中加入新模型后可直接调用

### <font color=violet>可用的弥散模板</font>
*    路径：    `/home/lhaaso/hushicong/Standard_prog_lib/Source_Analysis/Space_energy_Joint_fitting/DGE_Template`
*    文件名
        | 描述 | 文件名 | 坐标系 |
        | :--------: | :--------: | :--------: |
        |   PLANCK卫星测量的dust柱密度模板   |    `gll_dust_process.root`  | 银道     |
        |~|`gll_dust_process.txt`|银道|
        |~|`gll_dust_process_Eqm.root`|赤道|

### <font color=violet>可用数据</font>
*    WCDA（WCDA重建数据中各个量的含义见 [WCDAData_readme](https://jupyter.ihep.ac.cn/s/Ui_IqDNaf)）
        | 数据版本 | 数据时段 | 活时间【天】 |$N_{\rm hit}$|探测器|
        | :--------: | :--------: | :--------: | :--------: |:---------:|
        | Mk     |   20210305-20220930 |  508  |60-2000; 6 bins|全阵列|
        | Mk     |   20210305-20230731 |  796  |60-2000; 6 bins|全阵列|
        | Cod    |   20210305-20240131 |  978  |30-2000; 7 bins|全阵列|
        | Cod    |   20210305-20240731 |  1136  |30-2000; 7 bins|全阵列|

*    KM2A （KM2A重建数据中各个量的含义见 [KM2AData_readme](https://jupyter.ihep.ac.cn/s/OdCSM6uKT)）
        | 选择条件 | 数据时段 |   活时间【天】 |探测器|
        | :--------: | :--------: | :--------: | :--------:|
        |   CPC cut   |   20210720-20220930 |  420   |全阵列|
        |   CPC cut   |   20210720-20230731 |  710   |全阵列|
        |   CPC cut   |   20210720-20240131 |  884   |全阵列|
        |   CPC cut   |   20210720-20240731 |  1065   |全阵列|
*    WCDA的Mk版本和Cod版本的主要区别在于角分辨（PSF）的提高，如下图所示。基于Cod版本数据，Crab的$60\le N_{\rm hit}\le 100$的显著性提高约60%，Crab的$N_{\rm hit}\ge60$的显著性提高约25%。
<img src="https://jupyter.ihep.ac.cn/uploads/80b5a886-73d6-4fb6-a2a1-35730dfcb45f.png" width=90%>


### <font color=violet>方法说明</font>
*    某个源或弥散成分的TS的计算方法：将其他成分的位置和形态参数固定，放开目标源的所有参数进行拟合得到TS0，将目标源去除拟合得到TS1，TS_Src = TS0-TS1
*    某个源或弥散成分的每个段的TS的计算方法：将所有成分除能谱归一化因子（这套程序中对应参数'F0'）之外全部固定拟合得到TS0，去除目标源得到TS1，TS_Bin=TS0-TS1
*    流强上限点的计算方法：基于参数的likelihood profile，具体为：若某个源在某个bin的TS_Bin<4，则利用Minuit中'MINOS'计算F0的上下误差，Flux_UL = 最优拟合值+上误差。程序给出的Flux_UL为95%上限。

## <font color=orange>使用说明</font>
### <font color=violet>程序路径</font>
`/home/lhaaso/hushicong/Standard_prog_lib/Source_Analysis/Space_energy_Joint_fitting/v0.99`
### <font color=violet>主程序</font>
```bash=
./Src_Main
     [ configfile : Fit.yaml ]           // 拟合过程配置文件
     [ configfile : ParInit.yaml ]       // 参数初始值配置文件
```
*    <font color=red>提醒</font>：ROI较大或ROI内源的数量较多时，可能会遇到本地运行被kill的情况（可能是因为时间过长或内存占用过大），可提交作业运行。提交作业的命令如下：
        ```bash=
        hep_sub -g lhaaso -mem 15000 Src_Main -argu Fit.yaml ParInit.yaml
            -g lhaaso  : 指定用户组
            -mem 15000 : 指定最大运行内存，可根据ROI或源的数量调整。15000对于大部分情况是足够的。大内存作业一般排队时间较长……
        ```
### <font color=violet><a name="ParInit"></a>输入</font>
*    `Fit.yaml`：拟合过程配置文件
        *    示例：`config/Example1_Crab/Fit.yaml`
        ```yaml=
        # 主程序Src_Main所在的目录
        # 将程序拷贝到自己目录下后，要把这里的'WorkDir'改成自己的目录！
        WorkDir: /home/lhaaso/hushicong/Standard_prog_lib/Source_Analysis/Space_energy_Joint_fitting/v0.99
        
        # 数据配置文件在WorkDir/config/Data目录下，其中包括天图文件、响应文件、模拟数据文件。可根据自己需要选择相应时段或坐标系的配置文件
        DataConfig: 
            WCDA: config/Data/WCDA/Cod/Data_20210305_20240731.yaml
            KM2A: config/Data/KM2A/Data_20210720_20240731.yaml
        
        # 天图的坐标系选择：0为赤道坐标系下的天图，1为银道坐标系下的。目前已不再提供银道坐标系下的天图，故只能选0
        CorOpt: 0
        
        # 分析所用的探测器及能段
        DataUsed:
            # 平滑背景，暂不支持，可设为任意值
            SmoothBkg: 1
            WCDA: 
                # 分析中是否用WCDA数据，0为不用，1为用
                Active: 1
                # WCDA选用的能段，能段编号从0开始，可根据数据配置文件中的'Nhit'选择所需的能段
                NbinUsed: [1, 5]
                # 对数据天图和响应文件重新分bin，暂不支持，可将'Active'设置为0
                ReBin: 
                    Active: 0
                    Rebin: [2, 2, 1]
            KM2A:
                # 分析中是否用KM2A数据，0为不用，1为用
                Active: 1
                # KM2A选用的能段，这里输入的数值为log10(energy/TeV)，KM2A最大可用范围为[0.6, 3.4]
                NbinUsed: [1.4, 3.0]
        
        # ROI设置
        ROI:
            # 提前产生好的ROI文件
            ROIfile: none 
            # ROI 区域，六个参数的含义分别如下
                # var0：ROI坐标系选择，0为赤道坐标系，即按给定的坐标在赤道坐标系下选择ROI；1为银道坐标系。此处的坐标系选择不必要和上述的'CorOpt'一致
                # var1：ROI形状：0为圆形，1为矩形
                # var2 && var3：若var1为0，则var2和var3分别为ROI中心的横纵坐标；若var1为1，则var2和var3分别为ROI X的最小值和最大值
                # var4 && var5：若var1为0，则var4和var5分别为ROI半径和Model半径；若var1为1，则var4和var5分别为ROI Y的最小值和最大值
            Include: [0, 0, 83.63, 22.02, 4, 6]
            # mask ROI内指定区域，可参考下图'ROI示例'
            Exclude:
                # 是否mask，0为不做mask，1为mask
                Active: 0
                # mask区域，五个参数的含义同上述'Inlcude'的后五个参数。
                Region: [0, 80.63, 22.02, 1, 1]
        # 注意：ROI'Exclude'中的坐标值对应的坐标系必须和ROI'Inlcude'的相同

        # 快速迭代选项，1为使用快速迭代，0为不使用。可省略，默认值为1
        FastIteration: 1
        # 拟合过程设置：以下选项，0为不拟合，1为拟合
        Fit:
            # 拟合参数最优值
            Fitting: 1
            # 拟合每个能段的流强点
            FluxPoint: 1
            # 拟合每个源或弥散成分的TS
            TS_Src: 1
            # 拟合每个源或弥散成分每个段的TS
            TS_Bin: 1
            # 拟合每个源或弥散成分每个段的流强上限
            FluxUL: 1
        # 注意：若要拟合'FluxUL'，必须同时将'TS_Bin'设置为1
        # Tips：除'FluxUL'依赖于'TS_Bin'之外，其他都是独立的，即可以只把其中任意一个设为1，其他设为0，只做相应的分析，从而可以这样操作：
            #只把Fitting设为1，其他设为0，拟合得到参数最优值后，将参数最优值作为初始值，把Fitting改为0，FluxPoint设为1，继续算源的能谱点
            #再例如，逐个加源迭代确定ROI内源的个数时，可以只将Fitting设为1，不去计算其他东西，从而节省时间
    
        # TSmap设置
        TSmap:
            # 是否拟合TS map
            Active: 0
            # 拟合TS map是WCDA的数据选项：
                # var0: 是否用WCDA
                # var1 && var2: WCDA选用的能段
            WCDA: [1, 1, 5]
            KM2A: [1, 1.4, 3.0]
            # 做第SrcID个源的TS map，按ParInit.yaml中的源的顺序，SrcID从0开始
                # 特殊情况：SrcID设为-1，结合Subtract
            SrcID: 0
            # 0或1：0是指做TS map时扣除其他所有源，只保留SrcID；1是指仅扣除SrcID
                # 特殊情况：SrcID设为-1，Subtract为0做残差图（扣除包括弥散在内的所有源），Subtract为1做数据的显著性图（不扣除任何源或弥散）
            Subtract: 0
            # 拟合TS map的脚本
            JOBScript: JOB_TS.sh
        # 关于做TSmap的说明：
            # 若需要拟合TS map，只需要按上述说明设置并运行主程序即可，主程序会自动提交TS map的作业。
            # 作业会在本配置文件最后一部分的'Output'的'Outdir'/TSmap下生成多个root文件，只需要用'hadd TSmap_TargetSource.root TSmap_*.root'即可得到完整的目标源的TS map
            # 如果在第一次做'Fit'中的分析时，想要同时做某个源的TS map，只需按上述说明设置即可；若'Fit'中的分析已完成，'fConExcess'和'fParResu'已生成，此时若想做TS map，可将'Fit'中的选项全部设置成0，以节省时间
        # 注意：
            # 'TSmap'中的SrcID编号从0开始，顺序和ParInit.yaml中相同，0对应第一个源，弥散伽马始终是最后一个
            # 若要拟合TS map，必须先生成本配置文件最后的'fParResu'和fConExcess'
            # 'TS map'设置中所用的能段仅可在'DataUsed'中所用的能段范围内自由选择
               
        # 输出设置，每个选项会在下一小节“输出”中详述
        Output:
            # 平滑得到每个源的显著性图并保存到文件，0为不计算显著性天图，1为计算
            DrawOpt: 1
            # 输出文件的路径为WorkDir/Outdir
            Outdir: Results/Crab
            # 将参数最优值输出到指定文件，若不需要输出，则设置成'none'
            fParResu: ParRes.yaml
            # 将上述'Fitting'拟合结果对应的卷积后的超出数天图输出到指定文件，若不需要输出，则设置成'none'
            fConExcess: ConExcess.root
        ```
        *    ROI示例
        <img src="https://jupyter.ihep.ac.cn/uploads/2759292b-2fb3-4c26-baf0-e816c924518e.png" width=60%>


*    <a name="ParInit"></a>`ParInit.yaml`：源或弥散成分定义配置文件
        *    示例：`config/Example1_Crab/ParInit.yaml`
        ```yaml=
        # 弥散成分的设置
        DGE:
            # 分析中是否加入弥散成分，0为不加入，1为加入
            Active: 1
            # 拟合弥散成分时是否卷积PSF，0为不卷积，1为卷积；暂不支持，可设置为任意值
            ConvoPSF: 1
            # 第一个弥散成分的设置
            Template0:
                # 成分名称
                Name: dust
                # 模板文件：可设置成'ISO'即各向同性模板，也可指定模板文件。指定模板文件可为ROOT文件或txt文件
                    # txt文件：须包含三列，ra(l)、dec(b)、density，且第一行应为X坐标的起点、bin数和bin宽，第二行应为Y坐标的起点、bin数和bin宽（可参考DGE_Template/gll_dust_process.txt）
                    # ROOT文件：须包含一个存有模板信息的TH2
                Tempfile: /home/lhaaso/hushicong/Standard_prog_lib/Source_Analysis/Space_energy_Joint_fitting/DGE_Template/gll_dust_process_Eqm.root
                # 模板histogram的名字
                    # 若模板文件为txt文件，则此处必须是hTXT
                    # 若模板文件为ROOT文件，则此处可为除hTXT之外的任何符合C++变量命名规范的字符串
                TempHist: [hTemp_ana]
                # pivot energy
                Epiv: 10
                # 能谱模型设置
                SEDModel:
                    # 能谱模型
                    type: LP
                    # 四个参数含义分别为：
                        # var0：F0的初始值
                        # var1 && var2：F0限制范围的下届和上届
                        # var3：是否固定，0为不固定，1为固定
                        # var4: F0的数量级F0_order，实际计算中能谱的归一化因子等于F0*F0_order
                    F0: [2.3, 0, 5, 0, 1.e-13]
                    alpha: [2.6, 2.0, 3.0, 0]
                    beta: [0.1, 0, 1.0, 0]

        # 源成分的设置
        SRC:
            # 是否调用LHAASO Catalog的结果，0为调用，1为不调用；正在添加
            UseCatalog: 0
            # 分析中是否加入源成分，0为不加入，1为加入
            Active: 1
            # 全局的pivot energy：若大于0，则所有源的pivot energy均设置为该值；若小于0，则每个源的pivot energy设置为各自的值
            Epiv: 10
            # 全局的参数状态，即是否固定所有源的相应参数：0为不固定，1为固定
            ParStatus:
                Position: 0
                F0: 0
                # 能谱参数中除F0之外的所有参数
                Index: 0
                # 形态参数
                MorPar: 0
            Src0: 
                Name: Crab
                Epiv: 10
                SEDModel:
                    type: LP
                    # 参数含义同前
                    F0: [8.23737, 0, 50, 0, 1.e-14]
                    alpha: [2.88655, 2.0, 4.0, 0]
                    beta: [0.07781, 0.0, 0.3, 0]
                MorModel:
                    type: Ext_gaus
                    ra: [83.62180, 79.63, 87.63, 0]
                    dec: [22.01300, 18.02, 26.02, 0] 
                    sigma: [0.000, 0, 0.2, 0]
            Src1: 
                Name: Halo
                Epiv: 10
                SEDModel:
                    type: PEC 
                    F0: [2.97422, 0, 50, 0, 1.e-14]
                    alpha: [1.82436, 1.0, 4.0, 0]
                    Ecut: [32.27556, 5, 200, 0]
                MorModel:
                    type: Ext_gaus
                    ra: [85.67054, 81.85, 89.85, 0]
                    dec: [23.17787, 19.21, 27.21, 0]
                    sigma: [1.01160, 0.5, 6.0, 0]
            # Link parameters
            Src2:
                Name: Halo_2
                Epiv: 10
                # 设置Link parameters，可省略
                LinkPars:
                    # link Src2的谱型参数至其他源，数字为linked到的源的编号。源的编号即ParInit.yaml中的源的顺序，从0开始
                    # 注意事项：
                        # 1）ParInit.yaml中link的源和被link的源的能谱模型必须相同
                        # 2）ParInit.yaml中link的源必须写在被link的源之后
                    SED: 1
                SEDModel:
                    type: PEC
                    F0: [0.56846, 0.00, 500.00, 0, 1.e-14]
                    alpha: [1.82436, 1.00, 4.00, 0]
                    Ecut: [32.27556, 0.00, 500.00, 0]
                MorModel:
                    type: Ext_gaus
                    ra: [85.48447, 79.63, 87.63, 0]
                    dec: [23.31043, 18.02, 26.02, 0]
                    sigma: [0.41935, 0.00, 3.00, 0]
            # 使用固定模板（'Ext_Temp'）的源的设置方式  
            Src3: 
                Name: Test
                Epiv: 10
                SEDModel:
                    type: PEC 
                    F0: [2.97422, 0, 50, 0, 1.e-14]
                    alpha: [1.46248, 1.0, 4.0, 0]
                    Ecut: [26.48279, 5, 200, 0]
                MorModel:
                    type: Ext_Temp
                    Tempfile: xxx.root
                    TempHist: [xxx]
            # EBL吸收/CMB吸收修正
            Src4:
                Name: Mkn421
                Epiv: 1
                # EBL/CMB光深文件，文件中需要两列内容：能量(E/D):光深(Tau/D)。可参考config/Tutorial/Example4_ExtraGal/ParInit.yaml
                GGAbs: /home/lhaaso/hushicong/Standard_prog_lib/Source_Analysis/Space_energy_Joint_fitting/EBL_Model/Saldana_lopez2021/TAU_z0.031.txt
                SEDModel:
                    type: LP
                    F0: [29.03979, 0.00, 500.00, 0, 1.e-12]
                    alpha: [2.73769, 2.00, 4.00, 0]
                    beta: [0.32449, 0.00, 1.00, 0]
                MorModel:
                    type: Point
                    ra: [166.09439, 162.63, 170.63, 0]
                    dec: [38.17424, 34.02, 42.02, 0]
        ```
        *    <font color=red>注意</font>
                *    `ParInit.yaml`中某个成分的能谱或形态参数顺序必须和`src/Src_MorModel.yaml`及`src/Src_SEDModel.yaml`中保持一致，例如`src/Src_SEDModel.yaml`中带指数截断的幂律谱'PEC'的参数顺序为'[F0, alpha, Ecut]'，则`ParInit.yaml`中，'alpha'的设置必须写'F0'和'Ecut'之间，不能写到Ecut之后
                *    将程序拷贝到自己目录下后，需要将`Fit.yaml`中的`WorkDir`改成自己的程序所在的目录
### <font color=violet>输出</font>
*    参数最优值 && TS值 && 流强点/流强上限（直接输出到屏幕或指定文件）
        *    示例：![](https://jupyter.ihep.ac.cn/uploads/eebb7729-0d55-4496-9d0e-a75588afee5a.png)


                *    示例中'Flux Point'第七个点及之后是KM2A的结果，这里给出了两组KM2A的能量值及对应的流强值：第一组能量点为每个能段的中点值（例如对于$1.2<log_{10}(\frac{E}{\rm{TeV}})<1.4$能段，中点值为$10^{1.3} \rm{TeV}$）；第二组是根据源的位置和能谱利用模拟数据计算得到的。<font color=red>建议</font>：两组结果都保留，据我了解，KM2A组里两种做法都有。
                *    若在'Fit.yaml'选择了'FluxUL'选项，则此处的'Ferr1'或者'Ferr'为0意味着该流强点为流强上限
                *    示例中STATUS为拟合结果的检查
                
                        | Item | STATUS | 描述 |
                        | :--------: | :--------: | :--------: |
                        | ParVal and ParErr/TS of Component | OK | 收敛且Error Matrix准确 |
                        | ~ | NotOK | 不收敛或Error Matrix不准确 |
                        |TS for each bin/Flux point|OK|收敛且Error Matrix准确|
                        |~|NotOK|不收敛或Error Matrix不准确|
                        |~|F0Limit|F0最优值到边界/F0_UL超出边界|
                *    示例中'Chi2/NDF of SED'表征拟合得到的能谱与能谱点之间的一致程度，可用来检查所用的能谱模型是否能较好地描述该源的能谱（这里流强上限点不计入Chi2和NDF）
                *    Crab能谱
                <img src="https://jupyter.ihep.ac.cn/uploads/07380bfc-c7e6-4312-883e-6424e5baf130.png" width=70%>

                
        *    若`Fit.yaml`中的'fParResu'不为'none'，则将参数最优值按照`ParInit.yaml`中的格式输出的指定文件
                *    示例：`Results/Crab/ParRes.yaml`

*    形态
        *    若`Fit.yaml`中的'DrawOpt'为1，则将每个段的数据的显著性天图和残差显著性天图输出到'Outdir'下的`DataSigMap.root`中，将三个能段的数据及残差的显著性天图和三个能段每个源的显著性天图输出到'Outdir'下的`CompSigMap.root`中
        *    若`Fit.yaml`中的'fConExcess'不为'none'，则将每个源每个段的卷积得到的超出数天图输出到指定文件，方便后续计算超出数的一维分布及每个源的TS map
        *    示例：`Results/Crab/CompSigMap.root  ConExcess.root  DataSigMap.root`
                *    Crab周围的数据显著性天图及残差显著性天图
        ![](https://jupyter.ihep.ac.cn/uploads/d1e65ef8-e062-4620-b50c-6ff45709fa24.png)
        ![](https://jupyter.ihep.ac.cn/uploads/194fe6cb-b5b6-45ba-acc5-dcca0c07577b.png)
                *    三个能段的数据和三个能段每个源的显著性天图
        ![](https://jupyter.ihep.ac.cn/uploads/cf006e93-9ec0-48e8-92f6-6ff569b7712c.png)
                *    Crab的TS map（WCDA data：202103-202209，KM2A data：202107-202209）
        ![](https://jupyter.ihep.ac.cn/uploads/c4641cdb-4ff8-4965-a236-af4ee115c124.png)
        
*    源的径向流强profile（surfuce brightness profile，SBP）
        *    采用Fit.yaml中TSmap部分的设置，拟合以目标源为中心一定半径内的流强profile。将一定半径内的数据以目标源为中心划分成环带，分别拟合每个环带的数据得到相应的流强，进而得到profile。结果以柱状图（TH1D）的形式存在输出文件（OutFile）里。
                ```bash=
                ./Src_GetSBP
                  [ configfile : Fit.yaml ]
                  [ Radius for SBP analysis : in degree ]   //  分析区域的半径，原则上不应大于ROI
                  [ Bin Width : in degree*degree ]         //  环带的宽度 
                  [ OutFile : xxx.root ]
                ```
        *    结果示例如下：横坐标为角距离的平方
        <img src="https://jupyter.ihep.ac.cn/uploads/baf31b6e-964e-4e02-b5b4-54165c8287e3.png" width=70%>

### <font color=violet>v0.99/Tools：一些小工具</font>
*    v0.99/Tools下的小工具的功能及使用方法可见v0.9/Tools/README。<font color=red>注意</font>：与运行主程序一样，使用前需先source本文档开头的软件环境。目前已有的工具如下
        ```bash=
        ROOT2FITs.py : 
            将Src_Main生成的显著性天图（CompSigMap.root、DataSigMap.root）或Src_TSmap生成的TS map转换成fits格式
    
        ROOT2TXT.py :
            将Src_Main生成的显著性天图（CompSigMap.root、DataSigMap.root）或Src_TSmap生成的TS map转换成txt格式，txt文件中存有三列：x:y:significance
    
        SigMap_Eqm2Gal :
            将赤道坐标系下的天图转到银道坐标系下
        ```
*    <font color=orange>有任何其他的需求，可是随时联系我</font>

### <font color=violet><a name="分析样例"></a>分析样例</font>
以Crab区域的分析为例
输出到屏幕：`./Src_Main config/Tutorial/Example1_Crab/Fit.yaml config/Tutorial/Example1_Crab/ParInit.yaml`
输出到指定文件：`./Src_Main config/Tutorial/Example1_Crab/Fit.yaml config/Tutorial/Example1_Crab/ParInit.yaml &> log.txt`
*    ROI
        ```yaml=
        ROI:
          ROIfile: none
          Inlcude: [0, 0, 83.63, 22.02, 4, 6]  // ROI为(ra=83.63, dec=22.02)周围半径4度的区域
          Exclude:
          Active: 0                            
            Region: [0, 80.63, 22.02, 1, 1]
        ```

### <font color=violet><a name="结果检查"></a>结果检查</font>
结果的检查分为三个层级：
*    第零级：拟合是否正常，可用上述的'STATUS'
*    第一级：目标源所用的能谱模型或空间模型是否为最优：可换用不同的模型根据TS判据（ΔTS>25）、AIC判据或BIC判据（ΔAIC<-10，ΔBIC<-4，AIC和BIC的定义及说明可参考[DOI 10.1111/j.1745-3933.2007.00306.x](https://academic.oup.com/mnrasl/article/377/1/L74/1210361?login=false)）得到最优模型，同时可参考：1）上述的`chi2/NDF of SED`；2）根据超出数的一维分布计算得到的`chi2/NDF of Morphology`（即将加入）；3）残差显著性的一维分布与标准正态分布的偏离程度
*    第二级：考虑到目前可用的模型数量有限，可能出现第一级检查得到的最优模型的chi2/ndf依然较大的情况，这时候可能需要考虑添加新的模型或增加新的成分

### <font color=violet><a name="使用建议"></a>使用建议</font>
复杂区域的分析策略及简要说明
![](https://jupyter.ihep.ac.cn/uploads/8300f641-ee71-4e37-875c-f03f8e1e8d9e.png)
![](https://jupyter.ihep.ac.cn/uploads/650c027f-823a-4ee9-9548-109fb017bc52.jpg)
1.   N+1个源的模型相对于N个源的模型，ΔTS超过25，可认为N+1个源的模型显著优于N个源的模型，详细说明可参考[1LHAASO Catalog](https://arxiv.org/abs/2305.17030)文章；
2.   解析复杂区域逐个加源时，建议每轮迭代新加入的源的空间模型可以设为高斯扩展`Ext_gaus`，同时能谱模型尝试用`power-law/log-parabola/power-law with Ecut`去比较以得到更优的对数据的描述。迭代结束确定源的个数之后，可采用更复杂的空间模型和能谱模型对目标源做更精细的分析；
3.   解析复杂区域逐个加源时，新加入的源的参数初值可设为：位置初值可设为残差图上显著性最高的位置、能谱指数初值可设为2.6（单独分析WCDA）和3（单独分析KM2A）、形态可选择高斯扩展'Ext_gaus'且扩展度初值可设置为0.2度；
4.   做多源迭代时，在`ParInit.yaml`中增加第N+1个源前，可将`ParInit.yaml`中前N个源的参数初值更新成程序输出的`ParRes.yaml`中N个源的最优结果；
5.   单独用WCDA分析时，能谱模型可用power-law；单独分析KM2A或联合分析时，需要考虑如log-parabola或power-law with Ecut等更复杂的能谱模型；
6.   分析时可多次尝试调整`F0_order`，以使得F0接近于1，以加速收敛；
7.   Pivot energy（能谱公式中的E0）是使得F0对谱指数依赖最小的能量点，可参考文章[DOI 10.3847/0004-637X/817/1/3](https://iopscience.iop.org/article/10.3847/0004-637X/817/1/3/pdf)中的相关描述。可以选择能谱点误差较小的能量处作为E0。

### <font color=red>注意事项</font>
*    上述配置文件`Fit.yaml`中所有选项均不可省略；
*    ROI'Exclude'中的坐标值对应的坐标系必须和ROI'Inlcude'的相同；
*    若要拟合'FluxUL'，必须同时将'TS_Bin'设置为1；
*    `ParInit.yaml`中某个成分的能谱或形态参数顺序必须和`src/Src_MorModel.yaml`及`src/Src_SEDModel.yaml`中保持一致：例如`src/Src_SEDModel.yaml`中带指数截断的幂律谱'PEC'的参数顺序为'[F0, alpha, Ecut]'，则`ParInit.yaml`中，'alpha'的设置必须写'F0'和'Ecut'之间，不能写到Ecut之后；
*    ROI包含ra=360度时，输出显著性天图会有问题，很快会解决掉；