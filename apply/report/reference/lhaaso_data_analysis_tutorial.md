#                                                                                                                                                  LHAASO Data Analysis Tutorial
# 1. Apply for the account
This software is exclusively available on the IHEP server cluster, so you must apply for an account to proceed with the analysis. For detailed application instructions, please refer to: http://afsapply.ihep.ac.cn/cchelp/zh/accounts/. 

# 2 Quickstart Guide
After you apply the account, you can go to the INK web page (https://ink.ihep.ac.cn/) to starting the analysis. The post-login interface is displayed as follows:
><center>
> <table>
> <td><img src= "https://jupyter.ihep.ac.cn/uploads/7accc9b0-43eb-405a-8993-29ddbc5a9ed3.png" width="600" height="300" align="bottom"></td>
>     </table>
> <center>The page of the INK </center>

After successfully opening  both the Vscode and RootBrowse applications, you should see two connections listed under 'My Link Jobs'.

Click the connect before vscode label, navigate to the directory /home/lhaaso/your_name/, create a folder named 'crab_guide', then enter the new directory.The interface is shown in follow figure. All files within the crab_guide folder are displayed in the left panel, while a termination window appears in the bottom-right corner. 

><center>
> <table>
> <td><img src="https://jupyter.ihep.ac.cn/uploads/480bf0ac-149b-4257-a1cd-b6452f954fc0.png" width="600" height="300" align="bottom"></td>
>     </table>
> <center>The page of your work dir  </center>
 
    
**To use the software, installation is by setting up the environment.**
    
### ==Source enviroment in termination window ：==    
```shell
source /afs/ihep.ac.cn/users/x/xishaoqiang/.bashrc_everyone   
```
    
A preliminary environment is available to accelerate our data analysis(.bashrc_everyone_fast).    

    
In order to a quickly study, we also provide a script in the path "/home/lhaaso/xishaoqiang/crab_guide/run.sh". You can copy this script to your analysis directory. The script recored the command of this guide. 
    

    
Currently, the vscode interface can not view the .root file, thus click the connect before the RootBrowse label, navigate to the directory "File system/home/lhaaso/your_name/crab_guide/"".The interface is shown in follow figure. 
Simultenously, we can use this interface to check the figure listed in .root file. The interface of RootBrowse is shown in follow figure.
    
><center>
> <table>
> <td><img src="https://jupyter.ihep.ac.cn/uploads/5db6309e-bd2a-4b12-9013-f01f9343b822.png" width="600" height="300" align="bottom"></td>
>     </table>
> <center>The page of your RootBrowe  </center>
 

The tutorial makes extensive use of .yaml format configuration files. Tool *tune_yaml* is to modify the configuration file in your terminal window. 


## 2.1 Select the data

### ==**Create a data selection configuration file：**==
```shell
touch bg.yaml             
tune_yaml bg.yaml set_key selection all_sky_map /home/lhaaso/xishaoqiang/lhaaso/data/pass4/z50/data.root ##Add the path of Sky Cube.
tune_yaml bg.yaml set_key selection roi_map roi_ccube.root ##Set out put path.
tune_yaml bg.yaml set_key selection roi_x_range [84,78.9,87.3] ## ROI RA range.
tune_yaml bg.yaml set_key selection roi_y_range [80,18.1,26.1] ## ROI DEC range.
tune_yaml bg.yaml set_key selection roi_e_range [17,0.0,3.4]  ## Energy Range.
tune_yaml bg.yaml set_key selection roi_mask_e [1.1,1.3]  ## Energy Range.
```

### ==**Check your configuration file:**==

The configuration file is shown as follow:
  ```yaml
selection:
  all_sky_map: /home/lhaaso/xishaoqiang/lhaaso/data/pass4/z50/data.root
  roi_map: roi_ccube.root
  roi_x_range: [84,78.9,87.3] # ra_bins,ra_min,ra_max; interval must be 0.1 degree
  roi_y_range: [80,18.1,26.1] # dec_bins,dec_min,dec_max; interval must be 0.1 degree
  roi_e_range: [17,0.0,3.4] #bins,log10(e_min),log10(emax); interval must be 0.2
  roi_mask_e: [1.1, 1.3] #exclude the two km2a bins.
  ```

### ==**Get the data for analysis by perfoming *gtselect* tool：**==

```shell
gtselect bg.yaml
```

**==Note==：roi_e_range 0-1.0 are for WCDA data; roi_e_range 1.0-3.4 for KM2A data。**

### ==**Check the data using RootBrowse：**==
The interface is as follow:
    
><center>
> <table>
> <td><img src="https://jupyter.ihep.ac.cn/uploads/bf4fbf83-283e-43f1-92f5-83791498f9f6.png" width="600" height="300" align="bottom"></td>
>     </table>
> <center>The page of your RootBrowe  </center>
 


The figure of roi_map_on8 and roi_map_bg8:
><center>
> <table>
> <td><img src="https://jupyter.ihep.ac.cn/uploads/802a1ebd-024c-405d-84ad-da7537484421.png" width="600" height="300" align="bottom"></td>
>     <td><img src="https://jupyter.ihep.ac.cn/uploads/d1571e71-e193-4b2d-b4e8-86c2fc196e45.png" width="600" height="300" align="bottom"></td>
>     </table>
> <center>Figure 1. on map（left）and bg map（right），just one energy band</center>
>   </center>

## 2.2 Finding sources using TS map

### ==**Set and check your configuration file：**==

```shell
tune_yaml bg.yaml add_source iso_bg  #add CR background;
tune_yaml bg.yaml add_source gll_bg #add GDE background;
tune_yaml bg.yaml set_key output_option gttsmap tsmap_folder bg_tsmap #set the output folder for TS map 
tune_yaml bg.yaml set_key output_option gttsmap tsmap_x_range [88,79.2,88.0] #set ra ragne;
tune_yaml bg.yaml set_key output_option gttsmap tsmap_y_range [80,18.0,26.0] #set dec range;
vim bg.yaml
```

The configuration file is shown as follow：

```yaml
selection:
  all_sky_map: /home/lhaaso/xishaoqiang/lhaaso/data/pass4/z50/data.root
  roi_map: roi_ccube.root
  roi_x_range: [84, 78.9, 87.3]
  roi_y_range: [80, 18.1, 26.1]
  roi_e_range: [17, 0.0, 3.4]
source_dict:
  iso_bg: ### Cosmic-ray residual background; 
    sed_model:
      sed_type: PL
      norm: [1, 0, 0, 1e-14]
      index: [2.7, 0, 0]
      E_0: 10
    spatial_model:
      src_map: iso_bg_srcmap.root
      spatial_type: iso_bg
  gll_bg:
    sed_model:
      sed_type: PL
      norm: [5.63, 0.9314, 0.9314, 1e-15]
      index: [2.82, 0.1032, 0.1032]
      E_0: 50
    spatial_model:
      src_map: dust_gll_bg_srcmap.root
      spatial_type: gll_bg
      template_cutting: /home/lhaaso/xishaoqiang/lhaaso/data/pass2/gll_dust.root
      template_h2d_name: gll_region
      template_root_path: /home/lhaaso/xishaoqiang/lhaaso/data/GDE_Template/gll_dust.root
output_option:
  gttsmap:  ### gttsmap configuration key.
    tsmap_folder: bg_tsmap ### This is the output folder
    tsmap_x_range: [88, 79.2, 88.0]
    tsmap_y_range: [80, 18.0, 26.0]
```

### ==**Get the significance map by using *gttsmap* tool：**==

```shell
gtsrcmap bg.yaml #
gttsmap bg.yaml #
```
==**Note: once you change the context listed in source_dict, you must use gtsrcmap tool before performing any other operations.**==

**We will find that a series of jobs will be submitted to the server, and if the server is not queued, the parallel run will be completed very quickly. We can view the job running status.**

### ==**Check the job status：**==

```
hep_q -u
```

> 屏幕打印：
>
> 88569954.0      xishaoqiang     07/09 12:09     0+00:00:00      I  0   0.0  run.sh  
> 88569961.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
> 88569963.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
> 88569964.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
> 88569965.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
> 88569966.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
> 88569969.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
> 88569970.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
> 88569971.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
> 88569973.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
> 88569974.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
> 88569975.0      xishaoqiang     07/09 12:10     0+00:00:00      I  0   0.0  run.sh  
>
> 80 jobs; 0 completed, 0 removed, 80 idle, 0 running, 0 held, 0 suspended 

**We can see that 80 jobs have been submitted, wait for some time, use hep_q -u command, when we find that the jobs are complete (0 jobs is displayed, it means run complete!) Then run the run.sh script under the bg_tsmap folder.**


><center>
> <table>
> <td><img src="https://jupyter.ihep.ac.cn/uploads/8a953de8-1ada-4f3b-88ca-511ac5a9dbc0.png" width="600" height="300" align="bottom"></td>
>     </table>
> <center>The page of your RootBrowe  </center>

    
### ==**Enter the following command when all of the jobs are completed：**==

```shell
cd bg_tsmap
./run.sh   
```

    
    
### **Instroduction of the files listed in bg_tsmap folder：**

**(1) all_bash.sh:**  the script to submit the job;

**(2) run.sh:**  after the job is completed, merge the bg_tsmap.root script and convert .root file  to .fits file  as follows:：

```bash
curPath=$(dirname $(readlink -f "$0"))# the server automatically recognizes the current absolute path
cd $curPath
merge_tsmap.py tsmap.yaml ### Merge results from parallel runs
root2fits.py bg_tsmap.root bg_tsmap.fits TSMap J2000 ##.root file to .fits file。
get_plot_map_yaml.py tsmap.yaml 0 0.2 ### Create ploting configuration file, i.e., plot_map.yaml file.
plot_map.py plot_map.yaml ### Employ plot_map.py to plot the TS map
location_newsource.py tsmap.yaml 5 2 7 ## Recongnize the peak TS location, where 5 represents sigma large than 5，2 and 7 represent the color bar rangeing from 2 sigma to 7 sigma。
```

 **(3) bg_tsmap.fits and bg_tsmap.root :**  TS map file;

**(4) tsmap.yaml:** configuration file to create TS map file ；

**(5) plot_map.yaml:** configuration file to plot the TS map； 

**(6) source_plot.txt**: all of the source from the catalogs, such as TeVcat，Snrcat,  Fermi-LAT GeV catalog,  pulsar catalog;

**(7) bg_tsmap_add.yaml**: location of the peak TS value.

**(8) two .png format figures**:

><center>
> <table>
> <td><img src="https://jupyter.ihep.ac.cn/uploads/c2d049e2-5aac-46cb-b111-deb205f9f6d1.png" width="600" height="250" align="bottom"></td>
>     </table>
<center>Figure 2. Signifcance map（left,bg_map.png）and（right,bg_tsmap_loacated_source.png) </center> 
</center>
    


### Warning:
1. For the Sky region |b|>$15^\circ$, our default GDE modeling is unavalible. Therefore, when conducting an analysis of regions located with  |b|>$15^\circ$，it is necessary to exclude the GDE from the $source\_dict$ used in the analysis currently.

## 2.3 Solve the sources in the region

This step is the key of our data analysis. Based on the figure above, we need to add the source from bg_tsmap_add.yaml to the configuration file. After that, we should repeat the second step to locate the other sources. However, this process is often not easy to complete automatically, and it often requires manual adjustments, especially for complex areas. Fortunately, with the release of the 1LHAASO catalog, we can now start directly from the catalog source table. For complex regions, the parameters of WCDA and KM2A sources listed in the 1LHAASO catalog frequently show inconsistencies, likely due to energy-dependent morphological evolution in most cases or due to the under esitimated GDE emission. We have now implemented an updated catalog analysis using a combined WCDA-KM2A procedure, resulting in the preliminary 2LHAASO catalog.
    

### ==**Add the catalog sources to configureation file：**==

```shell
cp bg.yaml src_v1.yaml #copy bg.yaml to add the aim source；
tune_yaml src_v1.yaml add_source roi 7 # add the 2LHAASO sources in configurateion；
tune_yaml src_v1.yaml set_source free_all_norm
tune_yaml src_v1.yaml set_source free_one_norm gll_bg
gtsrcmap src_v1.yaml
gtlike src_v1.yaml src_v1.yaml
tune_yaml src_v1.yaml rm_source ts_cut 9 ## Based on ts value, get the source list which must be included in model file.
```

```yaml
selection:
  all_sky_map: /home/lhaaso/xishaoqiang/lhaaso/data/pass4/z50/data.root
  roi_map: roi_ccube.root
  roi_x_range: [84, 78.9, 87.3]
  roi_y_range: [80, 18.1, 26.1]
  roi_e_range: [17, 0.0, 3.4]
source_dict:
  iso_bg:  ### Cosmic-ray residual background;
    sed_model:
      sed_type: PL
      norm: [1, 0, 0, 1e-14]
      index: [2.7, 0, 0]
      E_0: 10
    spatial_model:
      src_map: iso_bg_srcmap.root
      spatial_type: iso_bg
  gll_bg: ### Diffuse Background considering plank dust distribution
    sed_model:
      norm: [5.4703, 0.9850, 0.9850, 1e-15]
      index: [2.82, 0.0, 0.0]
      E_0: 50
      sed_type: PL
    spatial_model:
      src_map: dust_gll_bg_srcmap.root
      spatial_type: gll_bg
      template_cutting: /home/lhaaso/xishaoqiang/lhaaso/data/pass4/z50/gll_dust.root
      template_h2d_name: gll_region
      template_root_path: /home/lhaaso/xishaoqiang/lhaaso/data/GDE_Template/gll_dust.root
    statistics:
      TS: 38.272
  J0534+2200:
    sed_model:
      sed_type: LP
      norm: [8.3690, 0.0255, 0.0255, 1e-14]
      index1: [2.8756, 0.0039, 0.0039]
      index2: [0.2014, 0.0045, 0.0045]
      E_0: 10
    spatial_model:
      src_map: J0534+2200_srcmap.root
      spatial_type: gaussian
      ra: [83.6344, 0.0010, 0.0010]
      dec: [22.0145, 0.0009, 0.0009]
      ext: [0.0386, 0.0027, 0.0027]
    statistics:
      TS: 274004.995
  J0543+2319:
    sed_model:
      sed_type: PLC
      norm: [3.2444, 0.0946, 0.0946, 1e-14]
      index: [1.5369, 0.0091, 0.0091]
      E_0: 10.0
      E_b: [25.2472, 0.1806, 0.1806]
    spatial_model:
      src_map: J0543+2319_srcmap.root
      spatial_type: two_gaussian
      ra: [85.9336, 0.0208, 0.0208]
      dec: [23.3101, 0.0148, 0.0148]
      ext: [1.2895, 0.0083, 0.0083]
      ra1: [85.8312, 0.0084, 0.0084]
      dec1: [23.2424, 0.0070, 0.0070]
      ext1: [0.9832, 0.0045, 0.0045]
      log_Erec_b: 1.1
    statistics:
      TS: 1818.2
output_option:
  gttsmap:
    tsmap_folder: bg_tsmap
    tsmap_x_range: [88, 79.2, 88.0]
    tsmap_y_range: [80, 18.0, 26.0]
```

The command "tune_yaml src_v1.yaml add_source roi 7" adds all 2LHAASO sources within a 7-degree region to the configuration file src_v1.yaml.

Note that:
1. The current 2LHAASO source catalog is preliminary and designed for combined fitting analysis
2. To specifically check 1LHAASO catalog sources, focus on individual components using:
（1）tune_yaml src_v1.yaml add_source roi 7 km2a (for KM2A-only sources)
（2）tune_yaml src_v1.yaml add_source roi 7 wcda (for WCDA-only sources)

Generally, we still need to optimize the source parameters in the configuration file and investigate whether there are new sources to be added to the model. There are only two sources in the Crab region. We can simply free all the parameters of the model to optimize the 1LHAASO catalog sources.  Note that the parameters may be not optimal by free all the parameters in most complex regions, dedicated tuning of the parameters are needed.
    
==**Enter the following command：**==
```shell
tune_yaml src_v1.yaml set_source free_all_sed_ext_p ###This command is to free all source spectrum and spatial parameters. Note that the two background parameters are still fixed, you can try to see how the configuration file changes after using this command.
tune_yaml src_v1.yaml set_source free_one_sed gll_bg ###This command is to free one source sed parameters. This command can free the galactic background This command can free the galactic background.
gtsrcmap src_v1.yaml ### calculate the source maps. 
gtlike src_v1.yaml src_v1.yaml #Fitting parameters;
gtsrcmap src_v1.yaml
gtlike src_v1.yaml src_v1.yaml #Fitting.
```


```yaml
### src_v1.yaml
selection:
  all_sky_map: /home/lhaaso/xishaoqiang/lhaaso/data/pass4/z50/data.root
  roi_map: roi_ccube.root
  roi_x_range: [84, 78.9, 87.3]
  roi_y_range: [80, 18.1, 26.1]
  roi_e_range: [17, 0.0, 3.4]
  roi_mask_e: [1.1, 1.3]
source_dict:
  iso_bg:
    sed_model:
      sed_type: PL
      norm: [1, 0, 0, 1e-14]
      index: [2.7, 0, 0]
      E_0: 10
    spatial_model:
      src_map: iso_bg_srcmap.root
      spatial_type: iso_bg
    each_bin:
      real_E: [0.15, 0.40, 0.60, 0.90, 1.25, 1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10, 3.30]
      real_E_error: [0.38, 0.33, 0.28, 0.25, 0.30, 0.16, 0.15, 0.13, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08]
      R_68: [0.50, 0.40, 0.33, 0.26, 0.19, 0.71, 0.57, 0.43, 0.36, 0.30, 0.26, 0.23, 0.20, 0.18, 0.17, 0.16, 0.15]
  gll_bg:
    sed_model:
      norm: [7.5711, 1.0686, 1.0686, 1e-15]
      index: [2.7716, 0.3199, 0.3199]
      E_0: 50
      sed_type: PL
    spatial_model:
      src_map: dust_gll_bg_srcmap.root
      spatial_type: gll_bg
      template_cutting: /home/lhaaso/xishaoqiang/lhaaso/data/pass4/z50/gll_dust.root
      template_h2d_name: gll_region
      template_root_path: /home/lhaaso/xishaoqiang/lhaaso/data/GDE_Template/gll_dust.root
    each_bin:
      real_E: [0.15, 0.40, 0.60, 0.90, 1.25, 1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10, 3.30]
      real_E_error: [0.38, 0.33, 0.28, 0.25, 0.30, 0.16, 0.15, 0.13, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08]
      R_68: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    statistics:
      TS: 51.4
  J0534+2200:
    sed_model:
      sed_type: LP
      norm: [8.2000, 0.0250, 0.0250, 1e-14]
      index1: [2.8683, 0.0008, 0.0008]
      index2: [0.1820, 0.0009, 0.0009]
      E_0: 10
    spatial_model:
      src_map: J0534+2200_srcmap.root
      spatial_type: gaussian
      ra: [83.6350, 0.0004, 0.0004]
      dec: [22.0143, 0.0004, 0.0004]
      ext: [0.0100, 0.0004, 0.0004]
    each_bin:
      real_E: [0.15, 0.40, 0.60, 0.90, 1.25, 1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10, 3.30]
      real_E_error: [0.38, 0.33, 0.28, 0.25, 0.30, 0.16, 0.15, 0.13, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08]
      R_68: [0.50, 0.40, 0.33, 0.26, 0.19, 0.71, 0.57, 0.43, 0.36, 0.30, 0.26, 0.23, 0.20, 0.18, 0.17, 0.16, 0.15]
    statistics:
      TS: 270477.1
  J0543+2319:
    sed_model:
      sed_type: PLC
      norm: [2.6820, 0.0786, 0.0786, 1e-14]
      index: [1.4720, 0.0117, 0.0117]
      E_0: 10.0
      E_b: [24.7717, 0.2145, 0.2145]
    spatial_model:
      src_map: J0543+2319_srcmap.root
      spatial_type: two_gaussian
      ra: [85.6917, 0.0419, 0.0419]
      dec: [23.2779, 0.0320, 0.0320]
      ext: [1.0789, 0.0172, 0.0172]
      ra1: [85.6634, 0.0177, 0.0177]
      dec1: [23.2098, 0.0148, 0.0148]
      ext1: [0.8572, 0.0096, 0.0096]
      log_Erec_b: 1.1
    each_bin:
      real_E: [0.15, 0.40, 0.60, 0.90, 1.25, 1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10, 3.30]
      real_E_error: [0.38, 0.33, 0.28, 0.25, 0.30, 0.16, 0.15, 0.12, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08]
      R_68: [0.50, 0.40, 0.33, 0.26, 0.19, 0.71, 0.56, 0.43, 0.36, 0.30, 0.26, 0.23, 0.20, 0.18, 0.17, 0.16, 0.15]
    statistics:
      TS: 1441.0
output_option:
  gttsmap:
    tsmap_folder: bg_tsmap
    tsmap_x_range: [84, 78.9, 87.3]
    tsmap_y_range: [80, 18.1, 26.1]
  gtlike:
    Error_status: 2
    negative_loglike: -139874.220
```
Please note that after using gtlike to fit all parameters specified in the configuration file, the Error_status value shown in output_option should be 3. If Error_status differs from 3, this indicates potential miscalculations in parameter errors. In such cases, we recommend recalculating parameter errors using the optimize_eachsrc tool, which estimates errors based on the hypothesis of source independence
. Even when Error_status equals 3, we still recommend verifying errors with optimize_eachsrc. Note that errors obtained by freeing all parameters will typically be slightly larger than those calculated by optimize_eachsrc


    
==**Enter the following command：**==
```shell
gtsrcmap src_v1.yaml
optimize_eachsrc src_v1.yaml src_v2.yaml 0 
gteachsrc_ts src_v2.yaml src_v2.yaml all 3  # fix the spatial model and the spectral shpae of each sources to re-calculate ts value.
```    

When comparing src_v2.yaml with srv1.yaml, as shown in Figure 3, we identified parameter discrepancies where src_v2.yaml contains the correct parameter values. Simultaneously, we observed smaller TS value in src_v2.yaml, which result from the gteachsrc_ts tool providing official TS values defined in the 1LHAASO catalog paper. These calculations follow the null hypothesis requirement to free the normalization of all background sources.
    
><center>
> <table>

> <td><img src="https://jupyter.ihep.ac.cn/uploads/50a2191a-49c4-4aa9-be40-3ec3a17ceb51.png" width="600" height="180" align="bottom"></td>
> <td><img src="https://jupyter.ihep.ac.cn/uploads/5272c661-e654-4b45-8bce-6d338c223d68.png" width="600" height="180" align="bottom"></td>
>     </table>
<center>Figure 3. Comparing src_v1.yaml and src_v2.yaml. </center> 
</center>
    


Based on the src_v2.yaml configuration file, we compute the tsmap to check if there are any other new sources that have not been discovered. We used get the minous TS to check the pixel distribution.

==**Enter the following command：**==

```shell
gtsrcmap src_v2.yaml 
tune_yaml src_v2.yaml set_tsmap 1 2.7 src_v2_tsmap
gttsmap src_v2.yaml
### when the job finish,we can get the residual map and the TS distribution use follow command.  
cd src_v2_tsmap
./run.sh
```

><center>
> <table>
> <td><img src="https://jupyter.ihep.ac.cn/uploads/c8ccb04c-ed9d-49a2-9b10-6cbe25b68acf.png" width="600" height="250" align="bottom"></td>
>     </table>
<center>Figure 4. Residual TS map subtructing all the sources in src_v2.yaml. </center> 
</center>
    
We need to verify whether the residual map follows a normal distribution. As shown in the right panel of the above figure, some excess components are present. Therefore, we should add a new source to the model file and repeat the previous analysis. In the src_v2_tsmap directory, we can find a file containing preliminary seed candidates. These seeds should be added to our source model for further iteration.

    
    
==**Enter the following command：**==

```shell
add_new_src src_v2.yaml src_v3.yaml src_v2_tsmap/bg_tsmap_add.yaml 
```
Two potential sources have been incorporated into our model. By default, the spectral energy distribution (SED) parameters are left free while the spatial model is fixed as a Gaussian distribution with 0.2 degree extension. See below:    
    
```yaml
### src_v3.yaml
selection:
  all_sky_map: /home/lhaaso/xishaoqiang/lhaaso/data/pass4/z50/data.root
  roi_map: roi_ccube.root
  roi_x_range: [84, 78.9, 87.3]
  roi_y_range: [80, 18.1, 26.1]
  roi_e_range: [17, 0.0, 3.4]
  roi_mask_e: [1.1, 1.3]
source_dict:
  iso_bg:
    sed_model:
      sed_type: PL
      norm: [1, 0, 0, 1e-14]
      index: [2.7, 0, 0]
      E_0: 10
    spatial_model:
      src_map: iso_bg_srcmap.root
      spatial_type: iso_bg
    each_bin:
      real_E: [0.15, 0.40, 0.60, 0.90, 1.25, 1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10, 3.30]
      real_E_error: [0.38, 0.33, 0.28, 0.25, 0.30, 0.16, 0.15, 0.13, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08]
      R_68: [0.50, 0.40, 0.33, 0.26, 0.19, 0.71, 0.57, 0.43, 0.36, 0.30, 0.26, 0.23, 0.20, 0.18, 0.17, 0.16, 0.15]
  gll_bg:
    sed_model:
      norm: [7.5711, 1.0686, 1.0686, 1e-15]
      index: [2.7716, 0.3199, 0.3199]
      E_0: 50
      sed_type: PL
    spatial_model:
      src_map: dust_gll_bg_srcmap.root
      spatial_type: gll_bg
      template_cutting: /home/lhaaso/xishaoqiang/lhaaso/data/pass4/z50/gll_dust.root
      template_h2d_name: gll_region
      template_root_path: /home/lhaaso/xishaoqiang/lhaaso/data/GDE_Template/gll_dust.root
    each_bin:
      real_E: [0.15, 0.40, 0.60, 0.90, 1.25, 1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10, 3.30]
      real_E_error: [0.38, 0.33, 0.28, 0.25, 0.30, 0.16, 0.15, 0.13, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08]
      R_68: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    statistics:
      TS: 51.4
  J0534+2200:
    sed_model:
      sed_type: LP
      norm: [8.2000, 0.0250, 0.0250, 1e-14]
      index1: [2.8683, 0.0008, 0.0008]
      index2: [0.1820, 0.0009, 0.0009]
      E_0: 10
    spatial_model:
      src_map: J0534+2200_srcmap.root
      spatial_type: gaussian
      ra: [83.6350, 0.0004, 0.0004]
      dec: [22.0143, 0.0004, 0.0004]
      ext: [0.0100, 0.0004, 0.0004]
    each_bin:
      real_E: [0.15, 0.40, 0.60, 0.90, 1.25, 1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10, 3.30]
      real_E_error: [0.38, 0.33, 0.28, 0.25, 0.30, 0.16, 0.15, 0.13, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08]
      R_68: [0.50, 0.40, 0.33, 0.26, 0.19, 0.71, 0.57, 0.43, 0.36, 0.30, 0.26, 0.23, 0.20, 0.18, 0.17, 0.16, 0.15]
    statistics:
      TS: 270477.1
  J0543+2319:
    sed_model:
      sed_type: PLC
      norm: [2.6820, 0.0786, 0.0786, 1e-14]
      index: [1.4720, 0.0117, 0.0117]
      E_0: 10.0
      E_b: [24.7717, 0.2145, 0.2145]
    spatial_model:
      src_map: J0543+2319_srcmap.root
      spatial_type: two_gaussian
      ra: [85.6917, 0.0419, 0.0419]
      dec: [23.2779, 0.0320, 0.0320]
      ext: [1.0789, 0.0172, 0.0172]
      ra1: [85.6634, 0.0177, 0.0177]
      dec1: [23.2098, 0.0148, 0.0148]
      ext1: [0.8572, 0.0096, 0.0096]
      log_Erec_b: 1.1
    each_bin:
      real_E: [0.15, 0.40, 0.60, 0.90, 1.25, 1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10, 3.30]
      real_E_error: [0.38, 0.33, 0.28, 0.25, 0.30, 0.16, 0.15, 0.12, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08]
      R_68: [0.50, 0.40, 0.33, 0.26, 0.19, 0.71, 0.56, 0.43, 0.36, 0.30, 0.26, 0.23, 0.20, 0.18, 0.17, 0.16, 0.15]
    statistics:
      TS: 1441.0
      J0543+2319:
    sed_model:
      sed_type: PLC
      norm: [2.6822, 0.0786, 0.0786, 1e-14]
      index: [1.4721, 0.1005, 0.1005]
      E_0: 10.0
      E_b: [24.7719, 2.1785, 2.1785]
    spatial_model:
      src_map: J0543+2319_srcmap.root
      spatial_type: two_gaussian
      ra: [85.6915, 0.1052, 0.1052]
      dec: [23.2778, 0.0797, 0.0797]
      ext: [1.0789, 0.0629, 0.0629]
      ra1: [85.6634, 0.0435, 0.0435]
      dec1: [23.2098, 0.0372, 0.0372]
      ext1: [0.8572, 0.0333, 0.0333]
      log_Erec_b: 1.1
    each_bin:
      real_E: [0.15, 0.40, 0.60, 0.90, 1.25, 1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10, 3.30]
      real_E_error: [0.38, 0.33, 0.28, 0.25, 0.30, 0.16, 0.15, 0.12, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08]
      R_68: [0.50, 0.40, 0.33, 0.26, 0.19, 0.71, 0.56, 0.43, 0.36, 0.30, 0.26, 0.23, 0.20, 0.18, 0.17, 0.16, 0.15]
    statistics:
      TS: 1441.0
  J0533+2227:
    sed_model:
      sed_type: PL
      norm: [1.0, 1e-5, 1e5, 1e-14]
      index: [3.0, 1, 6]
      E_0: 10
    spatial_model:
      src_map: J0533+2227_srcmap.root
      spatial_type: gaussian
      ra: [83.450000, 0, 0]
      dec: [22.450000, 0, 0]
      ext: [0.200000, 0, 0]
  J0536+2145:
    sed_model:
      sed_type: PL
      norm: [1.0, 1e-5, 1e5, 1e-14]
      index: [3.0, 1, 6]
      E_0: 10
    spatial_model:
      src_map: J0536+2145_srcmap.root
      spatial_type: gaussian
      ra: [84.150000, 0, 0]
      dec: [21.750000, 0, 0]
      ext: [0.200000, 0, 0]
output_option:
  gttsmap:
    tsmap_folder: bg_tsmap
    tsmap_x_range: [84, 78.9, 87.3]
    tsmap_y_range: [80, 18.1, 26.1]
  gtlike:
    Error_status: 2
    negative_loglike: -139874.220

```
==**Enter the following command to have a optimize：**==    
```shell
gtsrcmap src_v3.yaml
gtlike src_v3.yaml src_v3.yaml
tune_yaml src_v3.yaml set_source re_scale
tune_yaml src_v3.yaml sort_source ts
optimize_eachsrc src_v3.yaml src_v3.yaml 0
tune_yaml src_v3.yaml rm_source ts_cut 25
    
gtsrcmap src_v3.yaml
optimize_eachsrc_sed src_v3.yaml src_v3.yaml all 
    
tune_yaml src_v3.yaml set_source free_all_sed_ext_p 
tune_yaml src_v3.yaml set_source free_one_sed gll_bg    
gtsrcmap src_v3.yaml
gtlike src_v3.yaml src_v3.yaml
    
gtsrcmap src_v3.yaml
optimize_eachsrc src_v3.yaml src_v3.yaml 0

gtsrcmap src_v3.yaml
gteachsrc_ts src_v3.yaml src_v3.yaml all 3
```    
    
To ensure fitting convergence and accuracy, we implement an automated process that first searches for optimal initial parameters, then releases all parameters for final optimization to obtain the best-fitting model within our Region of Interest (ROI). This adaptive procedure requires strategy adjustments based on specific source environment characteristics.
The proposed strategy generally performs well in most cases. However, for the two sources adjacent to the Crab Nebula, the observed anomalies may stem from either imperfect PSF modeling or genuinely peculiar physical phenomena. Consequently, parameter estimation fails to converge for these particular sources.
 
==**Enter the following command to check residual map：**==
```shell
gtsrcmap src_v3.yaml 
tune_yaml src_v3.yaml set_tsmap 1 2.7 src_v3_tsmap
gttsmap src_v3.yaml
### when the job finish,we can get the residual map and the TS distribution use follow command.  
cd src_v3_tsmap
./run.sh
```      
><center>
> <table>
> <td><img src="https://jupyter.ihep.ac.cn/uploads/035dc965-d670-45f7-92a5-50eea570a131.png" width="600" height="250" align="bottom"></td>
>     </table>
<center>Figure 5. Residual TS map subtructing all the sources in src_v3.yaml. </center> 
</center>
    
All sources have now been successfully subtracted, yielding an clean residual map. For this ROI, the source modeling analysis has been completed.
    
    
    
## 2.4：Get SED

Although the information of the broad band spectrum is known in the configuration file, we still need to make the spectrum points to do multi-wavelength association. We also need to  check whether the model in the file is suitable for comparison between the spectrum points and the spectral model .

### ==Enter the following command：==

```shell
tune_yaml src_v3.yaml set_key output_option gtsed sed_folder "all_sed" #设置并行sed的文件夹all_sed；
tune_yaml src_v3.yaml set_key output_option gtsed e_bin_set "[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4]" #设置能谱点分bin； 
gtsed src_v3.yaml all #生成能谱的并行脚本并记录在all_sed文件夹中；
```

**Then we will find that there will be a series of jobs being submitted to the server, similar to that  gttsmap tool. After the jobs are finished, we can run the run.sh script under the folder, and then view the folder, we find the following:：
    

Note：

**sed_J0534+2200.txt  and  sed_J0542+2311.txt:** SED points in here；

**sed_broad.yaml:** Broad band spectrum, just a copy of src_v2.yaml；



<center>
<table>
<td><img src="https://jupyter.ihep.ac.cn/uploads/8491ac8c-121b-4bab-bebc-9270b5af3f35.png" width="600" height="300" align="bottom"></td>
</table>
Figure 2.SED 
</center>

But for other two sources, the observed spectrum exhibits anomalous features that may indicate potential mismodeling or systematic uncertanties.


# <font color=blue>3.  Introduction of the Parameter listed in Configuration File </font>

## 3.1 selection Parameter
The configure file parameter as following:
```
selection:
  all_sky_map: /home/lhaaso/xishaoqiang/lhaaso/data/pass3/data.root
  roi_map: roi_ccube.root
  roi_x_range: [84,78.9,87.3] # ra_bins,ra_min,ra_max; interval must be 0.1 degree
  roi_y_range: [80,18.1,26.1] # dec_bins,dec_min,dec_max; interval must be 0.1 degree
  roi_e_range: [17,0.0,3.4] #bins,log10(e_min),log10(emax); interval must be 0.2
```
### <font color=violet>The parameters for selection:</font>
*    Parameters table                         
        | Parameters | Values| comments |
        | :--------: | :-------: | :--------: |
        | all_sky_map|pass3/data.root;<br>pass4/z50/data.root;<br> pass4_full/data.root;<br>pass5/z50/data.root;|  The complete path for the all-sky data is detailed in the following section.  |
        | roi_map    |roi_ccube.root |  The output path of the selected data. |
        | roi_x_range   |[x_bins,x_min,x_max] |This parameter defines the rectangular boundary range in right ascension (R.A.) or galactic longitude（GLON） coordinates for the region of interest（ROI）.|
        | roi_y_range   |[y_bins,y_min,y_max] |This parameter defines the  boundary range in declination（Dec.） or galactic latitude（GLAT） coordinates for the ROI.|
        | roi_e_range   |[e_bins,e_min,e_max] |This parameter defines the  boundary range in  energies.|
        | roi_mask_e   |[mask1,mask2,...] |To exclude specific energy bins, enter their median values in brackets (e.g., setting [1.1,1.3] masks the first two KM2A energy bins)."|
        | roi_mask_map   | mask.root |To modify the ROI shape, create a mask.root file with binary values (0 for masked regions, 1 for selected areas) matching the map size defined above rectangular boundary.|
        | roi_coor_type   |J2000; galactic |This parameter defines the  boundary range in J2000 or galactic coordinate system|

### <font color=violet>The Avilable data:</font>
*    all_sky_map: /home/lhaaso/xishaoqiang/lhaaso/data/pass5/z50/data.root                     
        | Detector | Time range | live time(day) |Nhit or log10(E_rec/TeV) |bins|roi_e_range|
        | :--------: | :--------: | :--------: | :--------: |:---------:|:---------:|
        | WCDA     |   2021/03/05-2025/07/31 |  1484  |30-2000|7|[7, -0.4, 1.0]|
      | KM2A(full)     |   2021/07/20-2025/07/31 |  1438  |1.0-3.4|12|[12, 1.0, 3.4]|
      | KM2A(3/4)     |   2020/12/01-2021/07/19 |  216  |1.0-3.4|12|[12, 1.0, 3.4]|
      | KM2A(1/2)     |   2019/12/27-2020/11/30 |  289  |1.0-3.4|12|[12, 1.0, 3.4]|
*    all_sky_map: /home/lhaaso/xishaoqiang/lhaaso/data/pass4/z50/data.root                     
        | Detector | Time range | live time(day) |Nhit or log10(E_rec/TeV) |bins|roi_e_range|
        | :--------: | :--------: | :--------: | :--------: |:---------:|:---------:|
        | WCDA     |   2021/03/05-2024/07/31 |  1137  |30-2000|7|[7, -0.4, 1.0]|
      | KM2A(full)     |   2021/07/20-2024/07/31 |  1064  |1.0-3.4|12|[12, 1.0, 3.4]|
      | KM2A(3/4)     |   2020/12/01-2021/07/19 |  216  |1.0-3.4|12|[12, 1.0, 3.4]|
      | KM2A(1/2)     |   2019/12/27-2020/11/30 |  289  |1.0-3.4|12|[12, 1.0, 3.4]|

*    all_sky_map: /home/lhaaso/xishaoqiang/lhaaso/data/pass4_full/data.root                     
        | Detector | Time range | live time(day) |Nhit or log10(E_rec/TeV) |bins|roi_e_range|
        | :--------: | :--------: | :--------: | :--------: |:---------:|:---------:|
        | WCDA     |   2021/03/05-2024/07/31 |  1137  |30-2000|7|[7, -0.4, 1.0]|
      | KM2A(full)     |   2021/07/20-2024/07/31 |  1064  |1.0-3.4|12|[12, 1.0, 3.4]|


*    all_sky_map: /home/lhaaso/xishaoqiang/lhaaso/data/pass3/data.root                     
        | Detector | Time range | live time(day) |Nhit or log10(E_rec/TeV) |bins|roi_e_range|
        | :--------: | :--------: | :--------: | :--------: |:---------:|:---------:|
        | WCDA     |   2021/03/05-2024/01/31 |  979  |30-2000|7|[7, -0.4, 1.0]|
      | KM2A(full)     |   2021/07/20-2024/01/31 |  884  |1.0-3.4|12|[12, 1.0, 3.4]|
      | KM2A(3/4)     |   2020/12/01-2021/07/19 |  216  |1.0-3.4|12|[12, 1.0, 3.4]|
      | KM2A(1/2)     |   2019/12/27-2020/11/30 |  289  |1.0-3.4|12|[12, 1.0, 3.4]|


#### <font color=red>Warning:</font> The WCDA's Mod data version, implemented from pass3 onward, features enhanced Point Spread Function (PSF) resolution and improved detection efficiency for high zenith angle sources, and represents our current standard data product.

## 3.2 source_dict

The source dictionary comprises various astronomical source models, including:
Cosmic Ray (CR) background model
Galactic Diffuse Emission (GDE) model
Point source model
Gaussian distribution model
Other relevant source models
Each source model is uniquely characterized by two fundamental components:
Spectral model (defining energy distribution)
Spatial model (defining morphology)
The detailed specifications are as follows:



**Our spatial model and spectral model options are still being updated, currently available models include the following, you can try.**

### ==**SED models：**==

```yaml
#Power Law SED model
    sed_model:
      sed_type: PL
      norm: [1.1613, 1e-5, 1e5, 1e-14]
      index: [3.0465, 1, 5]
      E_0: 20
#Log-Parabola SED model
    sed_model:
      sed_type: LP
      norm: [5.2373, -0.3041, 0.3030, 1e-15]
      index1: [2.0172, -0.1632, 0.1507]
      index2: [2.0149, -0.2659, 0.2878]
      E_0: 20
#Power Law expcutoff SED model
    sed_model:
      sed_type: PLC
      norm: [1, 0, 0, 1e-14]
      index: [3, 0, 0]
      E_b: [40.3,0,0]
      E_0: 20
      beta: 1 ## the default value is 1.
#Broken Power Law SED model:
    sed_model:
      sed_type: BPL
      norm: [0.1707, 0.01937, 0.01937, 1e-17]
      index1: [2.1832, 0.17256, 0.17256]
      index2: [4.1925, 0.17096, 0.17096]
      E_b: [250.4132, 0, 0]
#Smooth Broken Power Law SED model:      
    sed_model:
      sed_type: SBPL
      norm: [0.1707, 0.01937, 0.01937, 1e-17]
      index1: [2.1832, 0.17256, 0.17256]
      index2: [4.1925, 0.17096, 0.17096]
      E_b: [250.4132, 0, 0]
      beta: 1
#IC CMB SED model：(Considering a EPLC electron spectrum)
    sed_model:
      sed_type: IC
      norm: [0.9988, 0.4645, 0.4645, 1e45] 
      index: [2.8795, 0.0868, 0.0868]
      E_b: [3000, 218.0425, 218.0425]
      E_0: 1.0 ## TeV
      d_pc: 1000 ## pc
      t_ph: 2.7  ## CMB T (K)
      edens_ph: 0.25 ### CMB density (eV)
#IC CMB and other taget photns SED model：(Considering a EPLC electron spectrum)
    sed_model:
      sed_type: IC
      norm: [0.9988, 0.4645, 0.4645, 1e45] 
      index: [2.8795, 0.0868, 0.0868]
      E_b: [3000, 218.0425, 218.0425]
      E_0: 1.0 ## TeV
      d_pc: 1000 ## pc
      t_ph: [2.7, 30, 5000]  ## T (K)
      edens_ph: [0.25,0.30,0.30] ###density (eV)
#Hadronic SED model：(Considering a EPLC proton spectrum)。   
    sed_model:
      sed_type: PP
      norm: [3.1678, 2.1211, 2.1211, 1e46]
      index: [1.9932, 0.1157, 0.1157]
      E_b: [2138.3607, 769.6618, 769.6618]
      E_0: 1.0
      d_pc: 1000 ## pc
      n_H: 1  ## target H density (1/cm3)
```
#### <font color=red>Warning:</font>: If you use  more taget photons for IC emission, you need to known the source distance. Considering a IRFS model, you can simpely find the density using the ISRF_phget.py tool.
==**Enter the following command：**==

```shell
ISRF_phget.py 272 -19.2 0.5 
```
where 272 is ra, -19.2 is dec, 0.5 is the distance of the source in kpc units.
### <font color=violet>SED Model Function</font>
*    Model Function defined as following:
        | Name   | flag in Configuration file |Function|
        | :--------: | :--------: | :--------: |
        |Power Law|PL|$$F=F_{0}\left(\frac{E}{E_0}\right)^{-\alpha}$$|
        |Log-parabola|LP|$$F=F_{0}\left(\frac{E}{E_{0}}\right)^{-\alpha_1-\alpha_2 {\rm log_{10}}\frac{E}{E_{0}}}$$|
        |Power Law expcutoff|PLC|$$F=F_{0}\left(\frac{E}{E_{\rm{0}}}\right)^{-\alpha}{\rm exp}^{-(\frac{E}{E_b})^{\beta}}$$|
        |Broken Power Law|BPL|$$F = F_{0}\left(\frac{E}{E_{\rm{b}}}\right)^{-\alpha_{1}} (E\le E_{\rm{b}}) \\ = \left(\frac{E}{E_{\rm{b}}}\right)^{-\alpha_{2}} (E\gt E_{\rm{b}})$$ |
        |Smooth Broken Power Law|SBPL|$$F=F_{0}\left(\frac{E}{E_{\rm{b}}}\right)^{-\alpha_{1}}\Bigg\{\frac{1}{2}\bigg[1+\left(\frac{E}{E_\rm{b}}\right)^{\frac{1}{\beta}}\bigg]\Bigg\}^{(\alpha_{1}-\alpha_{2})\beta}$$|
Note:  in configuration file, <font color=red> $F0$==norm;   $\alpha==index$; $\alpha1==index1$;$\alpha2==index2$; $\beta==$beta</font>

#### **==Spatial models：==**

```yaml
#point-like spatial model
    spatial_model:
      src_map: crab_srcmap.root
      spatial_type: ps
      ra: [83.6205, 82.6, 84.6]
      dec: [22.0356, 21.03, 23.03]
#Gaussian spatial model
    spatial_model:
      src_map: J0542+2311_srcmap.root
      spatial_type: gaussian 
      ra: [85.5090, 0.0874, 0.0874]
      dec: [23.0482, 0.0698, 0.0698]
      ext: [1.0699, 0.0612, 0.0612]
#Disk spatial model
    spatial_model:
      src_map: J0542+2311_srcmap.root
      spatial_type: disk 
      ra: [85.5090, 0.0874, 0.0874]
      dec: [23.0482, 0.0698, 0.0698]
      ext: [1.0699, 0.0612, 0.0612]
#Halo-like spatial model
    spatial_model:
      src_map: J0542+2311_srcmap.root
      spatial_type: halo
      ra: [85.5090, 0.0874, 0.0874]
      dec: [23.0482, 0.0698, 0.0698]
      ext: [1.0699, 0.0612, 0.0612]
#Ellipse disk sptial model
    spatial_model:
      src_map: J0542+2311_srcmap.root
      spatial_type: ellipse_disk 
      ra: [85.5090, 0.0874, 0.0874]
      dec: [23.0482, 0.0698, 0.0698]
      a_deg: [1.0699, 0.0612, 0.0612]
      b2a: [0.1, 0.0, 0.0]
      alpha: [30, 0.0, 0.0]
#Ellipse gaussian spatial model
    spatial_model:
      src_map: J0542+2311_srcmap.root
      spatial_type: ellipse_gauss
      ra: [85.5090, 0.0874, 0.0874]
      dec: [23.0482, 0.0698, 0.0698]
      a_deg: [1.0699, 0.0612, 0.0612]
      b2a: [0.1, 0.0, 0.0]
      alpha: [30, 0.0, 0.0]
#Rectangle spatial model 
    spatial_model:
      src_map: J0206+4307_rec_srcmap.root
      spatial_type: rectangle
      ra: [31.6266, 0.0853, 0.0853]
      dec: [43.1250, 0.0003, 0.0003]
      a_deg: [2.4172, 0.0147, 0.0147]
      b2a: [0.0742, 0.0007, 0.0007]
      alpha: [14.1273, 0.0588, 0.0588]
#two gaussian model, considering the different size for wcda and km2a.
    spatial_model:
      src_map: J0542+2311_srcmap.root
      spatial_type: two_gaussian
      ra: [287.0785, 0.0090, 0.0090]
      dec: [6.2948, 0.0093, 0.0093]
      ext: [0.3913, 0.0066, 0.0066]
      ra1: [287.0498, 0.0082, 0.0082]
      dec1: [6.2307, 0.0082, 0.0082]
      ext1: [0.3685, 0.0066, 0.0066]
      log_Erec_b: 1.1
#Other shape using a th2d expression
    spatial_model:
      src_map: file_bg_srcmap.root
      spatial_type: file_map
      template_h2d_name: file_th2d_name 
      template_root_path: file_th2d.root
```

# <font color=blue>4.  Introduction of Tool to tune the Parameter listed in Configuration File </font>

## ==tool1: tune_yaml：==

​        **You can see that in our analysis process, we frequently use the tool to adjust some parameters in the yaml configuration file.   These parameters can also be manually set. Because the current configuration file introduction is not perfect, there are many hidden features will be  in the future release. 。**

1. Fix all of the parameters listed in configuration file src.yaml: 
   ```shell 
   tune_yaml src.yaml set_source fixed_all
2. Free the norms of all the sources:
   ```shell 
   tune_yaml src.yaml set_source free_all_norm
3. Free  the norms, extensions of all the sources:
   ```shell 
   tune_yaml src.yaml set_source free_all_norm_ext
4. Other parameter setting (test by yourself):
   ```shell 
   tune_yaml src.yaml set_source free_all_sed
   tune_yaml src.yaml set_source free_all_sed_ext
   tune_yaml src.yaml set_source free_all_norm_ext_p
   tune_yaml src.yaml set_source free_all_sed_ext_p
5. Free sources listed in the 0.5 degree circle cenetered at (83,22):
   ```shell 
   tune_yaml src.yaml set_source free_roi 82 22 0.5
6. Free sources with extension larger than 0.5 degree:
     ```shell 
     tune_yaml src.yaml set_source free_ext_larger 0.5
7. Sort the sources listed in a configuration file:
   ```shell
   tune_yaml src.yaml sort_source ts ## accoring to significance
   tune_yaml src.yaml sort_source ext ## according to extension
   tune_yaml src.yaml sort_source ra 
   tune_yaml src.yaml sort_source l
   tune_yaml src.yaml sort_source b

## ==tool2: gtlikeProf：==
In consideration of potential errors in parameter estimation during the fitting process, we have developed a tool called gtlikeProf to examine the likelihood profile of individual parameters. Additionally, this tool allows for parallel fitting by fixing a specific parameter.
To use gtlikeProf command, you need set the configuration file in output_option key dictionary, like this :
```yaml
output_option:
  gtlikeProf:
    para_folder: J0544_a_deg ##The folder name
    source_name: J0544+2311  ## the source name
    para_name: a_deg  ## the parameter of the source
    para_bin: [41, 0.5, 2.5] ## the bins of the parameter. here is 41 bins from 0.5 degree to 2.5 degree.
```
For instance, when dealing with slow ellipse Gaussian morphology fitting, a strategy could involve sampling the extension and conducting the fitting as outlined below.

==**Enter the following command：**==

```shell
cp src_v2.yaml src_ellipse.yaml
tune_yaml src_ellipse.yaml gauss2ellipse J0544+2311
tune_yaml src_ellipse.yaml set_source free_one_sed_ext_p J0544+2311
tune_yaml src_ellipse.yaml set_key output_option gtlikeProf para_folder J0544_a_deg
tune_yaml src_ellipse.yaml set_key output_option gtlikeProf source_name J0544+2311
tune_yaml src_ellipse.yaml set_key output_option gtlikeProf para_name a_deg 
tune_yaml src_ellipse.yaml set_key output_option gtlikeProf para_bin [41,0.5,2.5] 
gtlikeProf src_ellipse.yaml
```
The "J0544_a_deg" folder will be created in your directory. You can review the contents of each folder within the "temp" directory.

## ==tool3: optimize_eachsrc：==
For complex regions characterized by an excess of five sources, achieving convergence proves challenging without an appropriate initial parameter configuration. To address this issue, we introduce a systematic tool designed to optimize each source individually while keeping the remaining sources fixed at their initial parameter values. This optimization process can be iteratively performed across multiple cycles to progressively refine the parameter estimates for all sources. It is important to note that during each iteration, the GDE model remains constrained and is not freely adjusted. In this approach, we first fit the normalization parameters for all sources, providing a robust starting point for subsequent iterative refinements.

==**Enter the following command：**==

```shell
cp src_v2.yaml src_v3.yaml
gtsrcmap src_v2.yaml
optimize_eachsrc src_v2.yaml src_v3.yaml 0 ## Here 0 is the optimize type, refer to following tabel.
tune_yaml src_v3.yaml set_source free_all_sed
tune_yaml src_v3.yaml set_source free_one_sed gll_bg
gtsrcmap src_v3.yaml
gtlike src_v3.yaml src_v3.yaml
```
### <font color=violet>Stratege of the optimize</font>
*    Optimize type defined as following:
        | flag in Configuration file |Description(set parameter of each sources in optimizeing|
        | :--------: | :--------: |
        |0|Free all the parameters|
        |1|Free SED parameters, extension parameters; |
        |2|Free SED parameters; |
        |3|Free norm parameter;|
        |5|Free norm parameter and spatial parameters;| 
        |6|Free norm parameter and extension parameters;|

## ==tool4: optimize_eachsrc_sed：==
Typically, we initially assume the spectral energy distribution (SED) type to be a power-law (PL). However, this assumption may not be suitable for all sources, particularly when performing broad-band spectral fitting involving more than 10 energy bins. To address this limitation, we provide a dedicated test tool to evaluate the  spectral shape of each source.We define the TS$_{curve}$=TS$_{PLC}$-TS$_{PL}$>6,we transition the spectral type from PL to PLC, ensuring a more accurate representation of the source's spectral characteristics.

<font color=red> if you want test all of the source listed in source_dict:</font>
==**Enter the following command：**==
```shell
gtsrcmap src_v3.yaml
optimize_eachsrc_sed src_v3.yaml src_v4.yaml all
```
<font color=red> if you want test one of the source (such as J0544+2311) listed in source_dict:</font>
==**Enter the following command：**==
```shell
gtsrcmap src_v3.yaml
optimize_eachsrc_sed src_v3.yaml src_v4.yaml J0544+2311
```
    
## ==tool5: optimize_eachsrc_ext：==
Typically, we initially assume the extension of sources is gaussian . However, this assumption may not be suitable for all sources. To address this limitation, we provide a dedicated test tool to evaluate the  extension significance.We define the TS$_{ext}$=TS$_{gauss}$-TS$_{ps}$<9, we transition the spatial type from gaussian to ps.

<font color=red> if you want test all of the source listed in source_dict:</font>
==**Enter the following command：**==
```shell
gtsrcmap src_v3.yaml
optimize_eachsrc_ext src_v3.yaml src_v4.yaml all
```
<font color=red> if you want to test one of the source (such as J0544+2311) listed in source_dict:</font>
==**Enter the following command：**==
```shell
gtsrcmap src_v3.yaml
optimize_eachsrc_ext src_v3.yaml src_v4.yaml J0544+2311
```
    
## ==tool6: gttsmap：==  
The tsmap tool is designed to generate TS maps by evaluating a point-like test source at each position. By default, the test source follows a Power-Law spectral shape with a fixed spectral index of 2.7. To modify the test source parameters, you need to specify a custom source model in the gttsmap options. 
<font color=red>To generate a TS map with the test source fixed at index 3.5,
==**Enter the following command：**==
```shell
tune_yaml src_v2.yaml set_tsmap 0 3.5 bg_tsmap
```
 You can get the configure file settings as follow:</font>
```yaml
output_option:
  gttsmap:
    tsmap_folder: bg_tsmap
    tsmap_x_range: [88, 79.2, 88.0]
    tsmap_y_range: [80, 18.0, 26.0]
    test_src:
      sed_model:
        sed_type: PL
        norm: [1.0, 1e-5, 1e5, 1e-15]
        E_0: 50
        index: [3.5, 0, 0]
      spatial_model:
        spatial_type: ps
```

<font color=red> By default, the TS map generation excludes negative TS values. To include negative TS values,
 ==**Enter the following command：**==
```shell
tune_yaml src_v2.yaml set_tsmap 1 2.7 residual_tsmap
```
You can get the configure file settings as follow:</font> 
    
```yaml
output_option:
  gttsmap:
    tsmap_folder: residual_tsmap
    tsmap_x_range: [88, 79.2, 88.0]
    tsmap_y_range: [80, 18.0, 26.0]
    test_src:
      sed_model:
        sed_type: PL
        norm: [1.0, -1e5, 1e5, 1e-15] 
        index: [2.7, 0.0, 0.0] 
        E_0: 50
      spatial_model:
        spatial_type: ps
```    
This configuration is commonly employed to examine the residual histogram upon completion of your analysis. </font> When configured this way, the system automatically generates residual histograms shell common in the run.sh in the "residual_tsmap" directory. 
    
<font color=red>This configuration allows testing any source model at different grid positions. For instance, to use a 1-degree Gaussian model as your test source, simply configure the settings as follows:: </font>
```yaml
output_option:
  gttsmap:
    tsmap_folder: bg_tsmap
    tsmap_x_range: [88, 79.2, 88.0]
    tsmap_y_range: [80, 18.0, 26.0]
    test_src:
      sed_model:
        sed_type: PL
        norm: [1.0, 1e-5, 1e5, 1e-15] 
        index: [3.5, 0.0, 0.0] 
        E_0: 50
      spatial_model:
        spatial_type: gaussian
        ext: [1.0,0,0]
```
## ==tool7: gtmodel：== 

This tool is designed to generate a count map for each source specified in the configuration file.
==**Enter the following command：**==
```shell
tune_yaml src_v2.yaml set_key output_option  residual_map "counts_map.root"
gtsrcmap src_v2.yaml 
gtmodel src_v2.yaml all
```
You can get the configure file settings as follow:</font> 
    
```yaml
output_option:
  residual_map: counts_map.root
```    
Then, the counts_map.root file will be generated in the current directory. Using ROOT, open this file to view the count map for each source specified in the configuration file as following:
><center>
> <table>
> <td><img src="https://jupyter.ihep.ac.cn/uploads/256eeef4-34ee-41b2-b751-02930cd6f50a.png" width="600" height="250" align="bottom"></td>
></table>
<center>Figure 7.Counts map for all the sources in src_v2.yaml. </center> 
</center>
In the counts_map.root file, each source's expected count map is stored in TH2D format, with the energy range consistent with that specified in the configuration file. 