require 'hand_parser'
class UploadsController < ApplicationController
    before_action :authenticate_user!
    include HandParser

    def new
        @upload = Upload.new
        hand_pos_name()
    end

    def create
        # check whether the upload file has been selected.
        if params[:upload][:avatar].nil?
            flash[:notice] = "Please choose your file to upload."
        else
            @upload = current_user.uploads.create(upload_params)
            if @upload.save
                flash[:notice] = "Successfully uploaded '#{@upload.avatar.filename}'."
            else
                flash[:notice] = "Only allow specific extensions ('mp4', 'avi', 'mov')."
            end
        end
        redirect_to controller: 'uploads', action: 'new', hand_pos: upload_params[:hand_pos]
    end

    def show
        if user_files_accessible_check
            @upload = Upload.find(params[:id])
        else
            flash[:notice] = "You are not allowed to operate this file."
            redirect_to status_path
        end
    end

    def destroy
        if user_files_accessible_check
            @upload = Upload.find(params[:id])
            @upload.destroy
            redirect_to controller: 'dashboard', action: 'index', hand_pos: @upload[:hand_pos]
        else
            flash[:notice] = "You are not allowed to operate this file."
            redirect_to status_path
        end
    end

    private
    def upload_params
        params.require(:upload).permit(:avatar, :hand_LR, :hand_pos)
    end
    
    def user_files_accessible_check
        @current_user_uploads_ids = current_user.uploads.ids
        return @current_user_uploads_ids.include? params[:id].to_i
    end
end
