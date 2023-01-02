class VideoToMp4Job
  include Sidekiq::Job

  def perform(upload_id)
    # Find video via upload id
    @upload = Upload.find(upload_id)
    video = @upload.video

    if video.attached? && video.blob.content_type != 'video/mp4'
  
      orig_video_tmpfile = "#{Rails.root}/tmp/#{video.blob.key}_#{video.blob.filename.to_s}"
      mp4_video_tmpfile = "#{Rails.root}/tmp/#{video.blob.key}_#{video.blob.filename.base}.mp4"
      File.open(orig_video_tmpfile, 'wb') do |f|
        f.write(video.download)
      end
  
      system('ffmpeg', '-i', orig_video_tmpfile, mp4_video_tmpfile)
  
      video.attach(
        io: File.open(mp4_video_tmpfile),
        filename: "#{video.blob.filename.base}.mp4",
        content_type: 'video/mp4'
      )
  
      File.delete(orig_video_tmpfile)
      File.delete(mp4_video_tmpfile)

    end
  end
end
