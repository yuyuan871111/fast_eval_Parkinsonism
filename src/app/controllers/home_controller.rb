class HomeController < ApplicationController
  def index
  end

  def documents
  end

  def demo
    @file = File.read("#{Rails.root}/public/demo/demo_left_normal_UPDRS_prediction.json")
    updrs_pred = JSON.parse(@file)
    @file = File.read("#{Rails.root}/public/demo/demo_left_normal_mp_hand_kpt_processed_handparams.json")
    hand_params = JSON.parse(@file)

    @demo_L = {
      :hand_LR => 'left',
      :err_frame_ratio => updrs_pred["error_frame_ratio"],
      :updrs => updrs_pred["predict_overall"],
      :mean_freq => hand_params["freq-mean"], 
      :std_freq => hand_params["freq-std"],
      :mean_inte => hand_params["intensity-mean"],
      :std_inte => hand_params["intensity-std"],
      :mean_inte_freq => hand_params["inte-freq-mean"],
      :std_inte_freq => hand_params["inte-freq-std"],
      :mean_peak => hand_params["peaks-mean"],
      :std_peak => hand_params["peaks-std"]
    }

    @file = File.read("#{Rails.root}/public/demo/demo_right_impaired_UPDRS_prediction.json")
    updrs_pred = JSON.parse(@file)
    @file = File.read("#{Rails.root}/public/demo/demo_right_impaired_mp_hand_kpt_processed_handparams.json")
    hand_params = JSON.parse(@file)

    @demo_R = {
      :hand_LR => 'right',
      :err_frame_ratio => updrs_pred["error_frame_ratio"],
      :updrs => updrs_pred["predict_overall"],
      :mean_freq => hand_params["freq-mean"], 
      :std_freq => hand_params["freq-std"],
      :mean_inte => hand_params["intensity-mean"],
      :std_inte => hand_params["intensity-std"],
      :mean_inte_freq => hand_params["inte-freq-mean"],
      :std_inte_freq => hand_params["inte-freq-std"],
      :mean_peak => hand_params["peaks-mean"],
      :std_peak => hand_params["peaks-std"]
    }
  end

  def status
  end
end
