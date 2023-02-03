class PyHandJob
  include Sidekiq::Job
  sidekiq_options retry: false
  require 'fileutils'
  require 'json'

  def perform(upload_id)
    # find upload file
    @upload = Upload.find(upload_id)
    @upload.update(status: 'running')
    video = @upload.video

    # set parameters
    wkdir_path = "#{Rails.root}/lib/hand_predictor"
    seed = "42"
    filename = "#{video.blob.filename.base}"
    ext = "mp4"
    hand_LR = @upload.hand_LR.to_s.capitalize
    hand_pos = @upload.hand_pos.to_s
    input_root_path = "#{Rails.root}/tmp/#{video.blob.key}"
    output_root_path = "#{input_root_path}/#{filename}"
    mode = "single"
    orig_video_tmpfile = "#{input_root_path}/#{filename}.#{ext}"
    py_script_path = "#{wkdir_path}/hand_predictor.py"

    # clean previous results (consider the retrying process in sidekiq)
    @upload.results.purge
    FileUtils.rm_rf(input_root_path)

    # write video from binary file
    FileUtils.mkdir_p(input_root_path)
    File.open(orig_video_tmpfile, 'wb') do |f|
      f.write(video.download)
    end

    begin
      # run analysis
      system("python", py_script_path, 
        "--wkdir_path", wkdir_path,
        "--seed", seed,
        "--filename", filename, 
        "--ext", ext, 
        "--hand_LR", hand_LR, 
        "--hand_pos", hand_pos,
        "--input_root_path", input_root_path, 
        "--output_root_path", output_root_path, 
        "--mode", mode)
      
      # compress the results in a zip file
      system("zip", "-qjr", "#{output_root_path}.zip", output_root_path)
      
      # check error frame ratio
      @file = File.read("#{output_root_path}/#{filename}_UPDRS_prediction.json")
      updrs_pred = JSON.parse(@file)
      @upload.updrs = updrs_pred["predict_overall"]
      @upload.err_frame_ratio = updrs_pred["error_frame_ratio"]
      
      # save blobs to database [zip, mp4, png, csv]
      blob_zip = ActiveStorage::Blob.create_and_upload!(io: File.open("#{output_root_path}.zip"), filename: "#{filename}.zip")
      blob_mp4_annot = ActiveStorage::Blob.create_and_upload!(io: File.open("#{output_root_path}/#{filename}_annot.mp4"), filename: "#{filename}_annot.mp4")
      if File.file?("#{output_root_path}/#{filename}_mp_hand_kpt.csv")
        blob_mpkpt = ActiveStorage::Blob.create_and_upload!(io: File.open("#{output_root_path}/#{filename}_mp_hand_kpt.csv"), filename: "#{filename}_mp_hand_kpt.csv")
      elsif File.file?("#{output_root_path}/#{filename}_mp_hand_kpt.thre0.csv")
        blob_mpkpt = ActiveStorage::Blob.create_and_upload!(io: File.open("#{output_root_path}/#{filename}_mp_hand_kpt.thre0.csv"), filename: "#{filename}_mp_hand_kpt.thre0.csv")
      else 
        blob_mpkpt = ActiveStorage::Blob.create_and_upload!(io: File.open("#{output_root_path}/#{filename}_mp_hand_kpt.empty.csv"), filename: "#{filename}_mp_hand_kpt.empty.csv")
      end
      blob_merge_png = @upload.err_frame_ratio < 1 ? ActiveStorage::Blob.create_and_upload!(io: File.open("#{output_root_path}/#{filename}_merge.png"), filename: "#{filename}_merge.png") : nil
      @upload.results.attach([blob_zip, blob_mp4_annot, blob_mpkpt, blob_merge_png])
      
      # save results to database
      if File.file?("#{output_root_path}/#{filename}_mp_hand_kpt_processed_handparams.json")
        @file = File.read("#{output_root_path}/#{filename}_mp_hand_kpt_processed_handparams.json")
      elsif File.file?("#{output_root_path}/#{filename}_mp_hand_kpt_processed.thre0_handparams.json")
        @file = File.read("#{output_root_path}/#{filename}_mp_hand_kpt_processed.thre0_handparams.json")
      else 
        @file = "{}"
      end
      hand_params = JSON.parse(@file)

      @upload.mean_freq = hand_params["freq-mean"] 
      @upload.std_freq = hand_params["freq-std"]
      @upload.mean_inte = hand_params["intensity-mean"]
      @upload.std_inte = hand_params["intensity-std"]
      @upload.mean_inte_freq = hand_params["inte-freq-mean"]
      @upload.std_inte_freq = hand_params["inte-freq-std"]
      @upload.mean_peak = hand_params["peaks-mean"]
      @upload.std_peak = hand_params["peaks-std"]
      @upload.save

      @upload.update(status: 'done')
    
    rescue => e
      Rails.logger.debug "[Hand_prediction error]:start-------"
      Rails.logger.debug e.message
      Rails.logger.error e.backtrace.join("\n")
      Rails.logger.debug "[Hand_prediction error]:end---------"
      @upload.update(status: 'fail')
    
    end
    
    # clean temp
    # File.delete(input_root_path)
    FileUtils.rm_rf(input_root_path)


  end
end
