require 'hand_parser'
class UploadsController < ApplicationController
    before_action :authenticate_user!
    include HandParser

    def new
        @upload = Upload.new
        hand_pos_name()
    end

    def create
        @upload = current_user.uploads.create(upload_params)
        if @upload.save
            flash[:notice] = "Successfully uploaded (#{@upload.avatar.filename})."
            #sleep(1)
            redirect_to status_path
        end
    end

    def destroy
        @upload.destroy
        redirect_to dashboard_index_path
    end

    private
    def upload_params
        params.require(:upload).permit(:avatar)
    end
end
