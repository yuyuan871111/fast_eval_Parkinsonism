require 'hand_parser'
class UploadsController < ApplicationController
    before_action :authenticate_user!
    include HandParser

    def new
        @upload = Upload.new
        hand_pos_name(params[:hand_pos])
    end

    def create
        # check whether the upload file has been selected.
        if params[:upload][:video].nil?
            flash[:notice] = "Please choose your file to upload."
        else
            @upload = current_user.uploads.create(upload_params)
            # byebug
            if @upload.save
                flash[:notice] = "Successfully uploaded '#{@upload.video.filename}'."
                VideoToMp4Job.perform_async(@upload.id)
            else
                flash[:notice] = "Only allow video extensions (e.g. 'mp4', 'avi', 'mov') and the maximum video size is 20MB."
            end
        end
        redirect_to controller: 'uploads', action: 'new', hand_pos: upload_params[:hand_pos]
    end

    def show
        if user_files_accessible_check
            @upload = Upload.find(params[:id])
            hand_pos_name(@upload.hand_pos)
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
            @hand_pos = @upload[:hand_pos]
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
            PyHandJob.perform_async(@upload.id)
            flash[:notice] = "The job is now queuing, come back to see results later."
            redirect_to controller: 'dashboard', action: 'index', hand_pos: @upload[:hand_pos]
        else
            flash[:notice] = "You are not allowed to operate this file."
            redirect_to status_path
        end
    end

    private
    def upload_params
        params.require(:upload).permit(:avatar, :hand_LR, :hand_pos, :video, :status)
    end

    def update_params
        params.require(:upload).permit(:hand_LR, :status)
    end
    
    def user_files_accessible_check
        @current_user_uploads_ids = current_user.uploads.ids
        return @current_user_uploads_ids.include? params[:id].to_i
    end

    rescue_from ActionController::Redirecting::UnsafeRedirectError do
        redirect_to status_path
    end
end
