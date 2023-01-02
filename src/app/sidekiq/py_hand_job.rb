class PyHandJob
  include Sidekiq::Job

  def perform(upload_id)
    @upload = Upload.find(upload_id)
    @upload.update(status: 'running')

    # orig_video_tmpfile = "#{Rails.root}/tmp/#{video.blob.key}_#{video.blob.filename.to_s}"
    # results_tmpfile = "#{Rails.root}/tmp/#{video.blob.key}_#{video.blob.filename.base}"
    # File.open(orig_video_tmpfile, 'wb') do |f|
    #   f.write(video.download)
    # end
    sleep 10

    @upload.update(status: 'done')
    @upload.update(retrieved_at: Time.current)
  end
end
