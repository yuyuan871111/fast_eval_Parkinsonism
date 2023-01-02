class PyHandJob
  include Sidekiq::Job

  def perform(upload_id)
    # find upload file
    @upload = Upload.find(upload_id)
    @upload.update(status: 'running')
    video = @upload.video

    # set parameters
    wkdir_path = "#{Rails.root}/lib/hand_predictor"
    seed = "42"
    filename = "#{video.blob.key}_#{video.blob.filename.base}"
    ext = "mp4"
    hand_LR = @upload.hand_LR.to_s.capitalize
    input_root_path = "#{Rails.root}/tmp"
    output_root_path = "#{Rails.root}/tmp/#{filename}"
    mode = "single"
    orig_video_tmpfile = "#{input_root_path}/#{filename}.#{ext}"
    py_script_path = "#{wkdir_path}/hand_predictor.py"

    # write video from binary file
    File.open(orig_video_tmpfile, 'wb') do |f|
      f.write(video.download)
    end

    # run analysis
    system("python", py_script_path, 
      "--wkdir_path", wkdir_path,
      "--seed", seed,
      "--filename", filename, 
      "--ext", ext, 
      "--hand_LR", hand_LR, 
      "--input_root_path", input_root_path, 
      "--output_root_path", output_root_path, 
      "--mode", mode)
    
    # clean temp
    File.delete(orig_video_tmpfile)

    @upload.update(status: 'done')
    @upload.update(retrieved_at: Time.current)
  end
end
