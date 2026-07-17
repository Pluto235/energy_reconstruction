#include "astro_fun.h"
#include "constants.h"
#include <iostream>
#include <string>
#include <TFile.h>
#include <TH2D.h>
#include "yaml.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include "TError.h"
#include <time.h>
#include <algorithm>
using namespace std;
static time_t tstart,tend;
static int i_time = 0;
static string out_yaml;

double lloglike_fun(const double *par){
    double f=0;/*{{{*/
    BinnedAnalysis &like = BinnedAnalysis::get_instance();
    Model_map& model_src = Model_map::get_instance();
    like.spatial_resrc_name.clear();

    for (unsigned i=0;i<like.sed_free_number;++i){
        //cout<<"i_sed:"<<i<<endl;
        like.src_config["source_dict"][like.sed_par_name[i].first]
            ["sed_model"][like.sed_par_name[i].second][0]=par[i];
    }

    for (unsigned i=0;i<like.spatial_free_number;++i){
        //cout<<"i_spatial:"<<(i+like.sed_free_number)<<endl;
        if (like.src_config["source_dict"][like.spatial_par_name[i].first]
            ["spatial_model"][like.spatial_par_name[i].second][0].as<double>()!=par[i+like.sed_free_number]){
            like.src_config["source_dict"][like.spatial_par_name[i].first]
            ["spatial_model"][like.spatial_par_name[i].second][0]=par[i+like.sed_free_number];
            if (find(like.spatial_resrc_name.begin(),like.spatial_resrc_name.end(),like.spatial_par_name[i].first)
                    ==like.spatial_resrc_name.end()){
                like.spatial_resrc_name.push_back(like.spatial_par_name[i].first);
            }
        }
    }
    for (vector<string>::const_iterator it=like.spatial_resrc_name.begin();
            it!=like.spatial_resrc_name.end();++it){
        model_src.change_one_srcmap(like.src_map_map[*it],*it, like.src_config);
    }

    for (vector<string>::const_iterator it=like.sed_resrc_name.begin();
            it!=like.sed_resrc_name.end();++it){
        like.set_src_sed(*it);
    }

    if (like.src_config["output_option"]["gtlike"]["tie_para"]){
        for(YAML::const_iterator it=like.src_config["output_option"]["gtlike"]["tie_para"].begin();it != like.src_config["output_option"]["gtlike"]["tie_para"].end(); ++it){/*{{{*/
            for (size_t x = 0; x<it->second.size();++x){
                if(x==0) continue;
                like.src_config["source_dict"][it->second[x][0].as<string>()]["sed_model"][it->second[x][1].as<string>()][0]=
                like.src_config["source_dict"][it->second[0][0].as<string>()]["sed_model"][it->second[0][1].as<string>()][0].as<double>();
                like.set_src_sed(it->second[x][0].as<string>());
            }
        }
    }

    like.get_like();
    f = like._loglike;
    //cout<<"likelihood is :"<<like._loglike<<endl;
    time(&tend);
    if((tend-tstart)/300>=1){
        like.src_config["output_option"]["gtlike"]["negative_loglike"]=to_string_wp(f,5);
        like.write_yaml(out_yaml);
        time(&tstart);
        cout<<"Saveing the temp fitting file with negative loglike: "<<to_string_wp(f,5)<<endl;
        i_time++;
        cout<<"Running time is about :"<<5*i_time<<" min;"<<endl;
    }
    return f;/*}}}*/
}

int main(int argc, char** argv){
    if(argc<3) {
       cout<<"Usage:"<<argv[0]<<" input.yaml output.yaml"<<endl;
       return 1;
    }/*}}}*/
    out_yaml = argv[2];


    time(&tstart);
    Model_map& model_src = Model_map::get_instance();
    YAML::Node config = YAML::LoadFile(argv[1]);
    model_src.setup(argv[1]);

    string minName = "Minuit"; 
    string algoName = "";
    ROOT::Math::Minimizer* minimum = ROOT::Math::Factory::CreateMinimizer(minName.c_str(), algoName.c_str());
    if (!minimum) {
       std::cerr << "Error: cannot create minimizer \"" << minName
                 << "\". Maybe the required library was not built?" << std::endl;
       return 1;
    }
    // set tolerance , etc...
    minimum->SetStrategy(2);
    minimum->SetMaxFunctionCalls(100000); // for Minuit/Minuit2
    minimum->SetMaxIterations(10000);  // for GSL
    if (config["output_option"]["gtlike"]["tol"]) minimum->SetTolerance(config["output_option"]["gtlike"]["tol"].as<float>());
    else minimum->SetTolerance(0.1);

    minimum->SetPrintLevel(1);


    BinnedAnalysis &like = BinnedAnalysis::get_instance();
    like.setup(argv[1]);
    like.load_srcmap();
    like.set_src_sed();
    cout<<"set_up is ok!"<<endl;
    like.get_free_sed_para_number();
    cout<<"get free sed number is "<<like.sed_free_number<<" !"<<endl;

    like.get_free_spatial_name();
    like.get_free_spatial_number();
    cout<<"get spatial free number is "<<like.spatial_free_number<<" !"<<endl;
    cout<<"likelihood is "<<to_string_wp(like._loglike,1)<<endl;

    
    //ROOT::Math::Functor f(&(BinnedAnalysis::loglike_fun),like.sed_free_number);
    int total_free_para = like.sed_free_number+like.spatial_free_number;
    if (total_free_para>=1){
        ROOT::Math::Functor f(&lloglike_fun,total_free_para);
        minimum->SetFunction(f);
        cout<<"math fun  is ok!"<<endl;

        string para_name;
        double para_init,para_up,para_low;
        double para_step = 0.005;
        double para_p_step = 0.01;
        for (unsigned i=0;i<like.sed_free_number;i++){
            para_name = like.sed_par_name[i].first+"_"+like.sed_par_name[i].second;
            para_init = like.src_config["source_dict"][like.sed_par_name[i].first]["sed_model"][like.sed_par_name[i].second][0].as<double>();
            para_low = like.src_config["source_dict"][like.sed_par_name[i].first]["sed_model"][like.sed_par_name[i].second][1].as<double>();
            para_up = like.src_config["source_dict"][like.sed_par_name[i].first]["sed_model"][like.sed_par_name[i].second][2].as<double>();
            if (para_init<para_low) para_init = sqrt(para_low*para_up);
            if (like.sed_par_name[i].second =="E_b"){
                minimum->SetLimitedVariable(i,para_name,para_init,0.5,para_low,para_up);
            }
            else{
                minimum->SetLimitedVariable(i,para_name,para_init,para_step,para_low,para_up);
            }
        }

        for (unsigned i=0;i<like.spatial_free_number;i++){
            para_name = like.spatial_par_name[i].first+"_"+like.spatial_par_name[i].second;
            para_init = like.src_config["source_dict"][like.spatial_par_name[i].first]["spatial_model"][like.spatial_par_name[i].second][0].as<double>();
            para_low = like.src_config["source_dict"][like.spatial_par_name[i].first]["spatial_model"][like.spatial_par_name[i].second][1].as<double>();
            para_up = like.src_config["source_dict"][like.spatial_par_name[i].first]["spatial_model"][like.spatial_par_name[i].second][2].as<double>();
            minimum->SetLimitedVariable(i+like.sed_free_number,para_name,para_init,para_p_step,para_low,para_up);
        }
        cout<<"para seting is ok!"<<endl;
        minimum->SetErrorDef(0.5);
        if (like.src_config["output_option"]["gtlike"]){
            if(like.src_config["output_option"]["gtlike"]["optimizer"]){
                like.optimizer=like.src_config["output_option"]["gtlike"]["optimizer"].as<string>();
            }
            if(like.src_config["output_option"]["gtlike"]["upper_limits"]){
                minimum->SetErrorDef(like.src_config["output_option"]["gtlike"]["upper_limits"].as<double>());
                like.optimizer="minos";
            }
        }
        minimum->Minimize();
        like.like_status =  minimum->CovMatrixStatus();
        if (like.like_status!=3) minimum->Hesse();

        double para[like.sed_free_number+like.spatial_free_number];
        double low_err[like.sed_free_number+like.spatial_free_number];
        double up_err[like.sed_free_number+like.spatial_free_number];

        if(like.optimizer=="minos"){
            for (unsigned i=0;i<like.sed_free_number+like.spatial_free_number;++i){
                minimum->GetMinosError(i,low_err[i],up_err[i]);
                para[i]= minimum->X()[i];
            }
        }
        else{
            for (unsigned i=0;i<like.sed_free_number+like.spatial_free_number;++i){
                para[i]= minimum->X()[i];
                low_err[i] = minimum->Errors()[i];
                up_err[i] = minimum->Errors()[i];
            }
        }
        like.like_status = minimum->CovMatrixStatus();
        cout<<"Error matrix status is:"<<like.like_status<<"; if it !=3 ,"<< 
        "there maybe an invalide in error of some parameters;"<<endl;
        like.src_config["output_option"]["gtlike"]["Error_status"]=like.like_status;
        like.src_config["output_option"]["gtlike"]["covariance_status"]=like.like_status;
        like.src_config["output_option"]["gtlike"]["edm"]=minimum->Edm();
        like.src_config["output_option"]["gtlike"]["minimum_value"]=minimum->MinValue();
        like.src_config["output_option"]["gtlike"]["function_calls"]=(unsigned long long)minimum->NCalls();

        YAML::Node covariance_names(YAML::NodeType::Sequence);
        for (unsigned i=0;i<like.sed_free_number;i++){
            covariance_names.push_back(like.sed_par_name[i].first+"_"+like.sed_par_name[i].second);
        }
        for (unsigned i=0;i<like.spatial_free_number;i++){
            covariance_names.push_back(like.spatial_par_name[i].first+"_"+like.spatial_par_name[i].second);
        }

        YAML::Node covariance(YAML::NodeType::Sequence);
        YAML::Node correlation(YAML::NodeType::Sequence);
        for (int i=0;i<total_free_para;i++){
            YAML::Node covariance_row(YAML::NodeType::Sequence);
            YAML::Node correlation_row(YAML::NodeType::Sequence);
            const double sigma_i = minimum->Errors()[i];
            for (int j=0;j<total_free_para;j++){
                const double value = minimum->CovMatrix(i,j);
                const double sigma_j = minimum->Errors()[j];
                covariance_row.push_back(value);
                correlation_row.push_back((sigma_i>0 && sigma_j>0) ? value/(sigma_i*sigma_j) : 0.0);
            }
            covariance.push_back(covariance_row);
            correlation.push_back(correlation_row);
        }
        like.src_config["output_option"]["gtlike"]["covariance_parameter_names"]=covariance_names;
        like.src_config["output_option"]["gtlike"]["covariance"]=covariance;
        like.src_config["output_option"]["gtlike"]["correlation"]=correlation;
        like.get_free_spatial_name();
        // 写参数文件。
        for (unsigned i=0;i<like.sed_free_number;i++){
            like.src_config["source_dict"][like.sed_par_name[i].first]
                ["sed_model"][like.sed_par_name[i].second][0]=to_string_wp(para[i],4);
            like.src_config["source_dict"][like.sed_par_name[i].first]
                ["sed_model"][like.sed_par_name[i].second][1]=to_string_wp(low_err[i],4);
            like.src_config["source_dict"][like.sed_par_name[i].first]
                ["sed_model"][like.sed_par_name[i].second][2]=to_string_wp(up_err[i],4);
        }
   
        for (unsigned i=0;i<like.spatial_free_number;i++){
            like.src_config["source_dict"][like.spatial_par_name[i].first]
                ["spatial_model"][like.spatial_par_name[i].second][0]=to_string_wp(para[i+like.sed_free_number],4);
            like.src_config["source_dict"][like.spatial_par_name[i].first]
                ["spatial_model"][like.spatial_par_name[i].second][1]=to_string_wp(low_err[i+like.sed_free_number],4);
            like.src_config["source_dict"][like.spatial_par_name[i].first]
                ["spatial_model"][like.spatial_par_name[i].second][2]=to_string_wp(up_err[i+like.sed_free_number],4);
        }

        if (like.src_config["output_option"]["gtlike"]["tie_para"]){
            for(YAML::const_iterator it=like.src_config["output_option"]["gtlike"]["tie_para"].begin();it != like.src_config["output_option"]["gtlike"]["tie_para"].end(); ++it){
                for (size_t x = 0; x<it->second.size(); ++x){
                    if(x==0) continue;
                    like.src_config["source_dict"][it->second[x][0].as<string>()]["sed_model"][it->second[x][1].as<string>()][0]=
                    to_string_wp(like.src_config["source_dict"][it->second[0][0].as<string>()]["sed_model"][it->second[0][1].as<string>()][0].as<double>(),4);

                    like.src_config["source_dict"][it->second[x][0].as<string>()]["sed_model"][it->second[x][1].as<string>()][1]=
                    to_string_wp(like.src_config["source_dict"][it->second[0][0].as<string>()]["sed_model"][it->second[0][1].as<string>()][1].as<double>(),4);

                    like.src_config["source_dict"][it->second[x][0].as<string>()]["sed_model"][it->second[x][1].as<string>()][2]=
                    to_string_wp(like.src_config["source_dict"][it->second[0][0].as<string>()]["sed_model"][it->second[0][1].as<string>()][2].as<double>(),4);
                }
            }
        }

        for (vector<string>::const_iterator it=like.spatial_resrc_name.begin();
                it!=like.spatial_resrc_name.end();++it){
            //model_src.get_one_srcmap(*it, like.src_config);
            model_src.change_one_srcmap(like.src_map_map[*it],*it, like.src_config);
        }
    }
    like.set_src_sed();
    like.get_like();
    cout<<"likelihood is "<<to_string_wp(like._loglike,5)<<endl;
    like.src_config["output_option"]["gtlike"]["negative_loglike"]=to_string_wp(like._loglike,5);

    // get the ts value for each source
    double srcTs;
    for(YAML::iterator it=like.src_config["source_dict"].begin();it != like.src_config["source_dict"].end(); ++it){
        if(it->first.as<string>()=="iso_bg") continue;
        srcTs = like.get_srcTS_fast(it->first.as<string>());
        like.src_config["source_dict"][it->first.as<string>()]["statistics"]["TS"] = to_string_wp(srcTs);
    }

    //for (vector<string>::const_iterator it=like.spatial_resrc_name.begin();
    //        it!=like.spatial_resrc_name.end();++it){
    //    model_src.get_one_srcmap(*it, like.src_config);
    //}

    like.get_src_real();
    like.write_yaml(out_yaml);
    return 0;
}
