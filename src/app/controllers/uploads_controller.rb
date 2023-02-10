require 'hand_parser'
class UploadsController < ApplicationController
    before_action :authenticate_user!
    include HandParser

    def new
        @upload = Upload.new
        hand_pos_name(params[:hand_pos])
        # check data usage
        @record_ids = ActiveStorage::Attachment.where(record: current_user.uploads).pluck(:record_id)
        current_user.data_usage = ActiveStorage::Blob.joins(:attachments).where(active_storage_attachments: { record_id: @record_ids}).sum(:byte_size)
        current_user.save
    end

    def create
        # check whether the upload file has been selected.
        if params[:upload][:video].nil?
            flash[:notice] = "Please choose your file to upload."
        elsif current_user.data_usage > current_user.data_usage_limit
            flash[:notice] = "You have reached the limit of data usage, please clean useless files or contact to us to enlarge your space."
        else
            @upload = current_user.uploads.create(upload_params)
            # byebug
            if @upload.save
                flash[:notice] = "Successfully uploaded '#{@upload.video.filename}'."
                VideoToMp4Job.perform_async(@upload.id)
            else
                flash[:notice] = "Only allow video extensions (e.g. 'mp4', 'avi', 'mov') and the maximum video size is 50MB."
            end
        end
        redirect_to controller: 'uploads', action: 'new', hand_pos: upload_params[:hand_pos]
    end

    def show
        if user_files_accessible_check
            @upload = Upload.find(params[:id])
            hand_pos_name(@upload.hand_pos)
            @empty_upload = {
                "updrs" => nil,
                "mean_freq" => nil,
                "mean_inte" => nil,
                "mean_inte_freq" => nil,
                "mean_peak" => nil
            }
        else
            flash[:notice] = "You are not allowed to view this file."
            redirect_to status_path
        end
    end

    def edit
        if user_files_accessible_check
            @upload = Upload.find(params[:id])
            hand_pos_name(@upload.hand_pos)
        else
            flash[:notice] = "You are not allowed to operate this file."
            redirect_to status_path
        end
    end

    def update
        if user_files_accessible_check
            @upload = Upload.find(params[:id])
            hand_pos_name(@upload.hand_pos)
            if @upload.status == "running" || @upload.status == "queuing"
                flash[:notice] = "This file is still processing, please wait a couple minutes."
                redirect_to status_path
            elsif @upload.update(update_params)
                reset_results(@upload)
                flash[:notice] = "Successfully updated."
                redirect_to controller: 'dashboard', action: 'index', hand_pos: @upload.hand_pos
            else
                redirect_to controller: 'uploads', action: 'edit', hand_pos: @upload.hand_pos
            end
        else
            flash[:notice] = "You are not allowed to operate this file."
            redirect_to status_path
        end
        
    end

    def destroy
        if user_files_accessible_check
            @upload = Upload.find(params[:id])
            @upload.destroy
            flash[:notice] = "Successfully deleted."
            redirect_to controller: 'dashboard', action: 'index', hand_pos: @upload[:hand_pos]
        else
            flash[:notice] = "You are not allowed to operate this file."
            redirect_to status_path
        end
    end

    def do_run
        if user_files_accessible_check
            @upload = Upload.find(params[:id])
            @upload.update(status: 'queuing')
            @upload.update(submitted_at: Time.current)
            PyHandJob.perform_async(@upload.id)
            flash[:notice] = "The job is now queuing, come back to see results later."
            redirect_to controller: 'dashboard', action: 'index', hand_pos: @upload[:hand_pos]
        else
            flash[:notice] = "You are not allowed to operate this file."
            redirect_to status_path
        end
    end

    def archive
        if user_files_accessible_check
            @upload = Upload.find(params[:id])
            @upload.update(:archived => true)
            flash[:notice] = "Successfully archived."
            redirect_to controller: 'dashboard', action: 'index', hand_pos: @upload[:hand_pos]
        else
            flash[:notice] = "You are not allowed to operate this file."
            redirect_to status_path
        end
    end

    def unarchive
        if user_files_accessible_check
            @upload = Upload.find(params[:id])
            @upload.update(:archived => false)
            flash[:notice] = "Successfully unarchived."
            redirect_to controller: 'dashboard', action: 'archive', hand_pos: @upload[:hand_pos]
        else
            flash[:notice] = "You are not allowed to operate this file."
            redirect_to status_path
        end
    end

    private
    def upload_params
        params.require(:upload).permit(:hand_LR, :hand_pos, :video, :status, :recorded_at)
    end

    def update_params
        params.require(:upload).permit(:hand_LR, :status, :recorded_at)
    end

    def reset_results(upload)
        upload.updrs = nil
        upload.err_frame_ratio = nil 
        upload.submitted_at = nil
        upload.mean_freq = nil
        upload.std_freq = nil
        upload.mean_inte = nil
        upload.std_inte = nil
        upload.mean_inte_freq = nil
        upload.std_inte_freq = nil
        upload.mean_peak = nil
        upload.std_peak = nil
        upload.save
        upload.results.purge
    end
    
    def user_files_accessible_check
        @current_user_uploads_ids = current_user.uploads.ids
        return @current_user_uploads_ids.include? params[:id].to_i
    end

    rescue_from ActionController::Redirecting::UnsafeRedirectError do
        redirect_to status_path
    end
end
