require 'hand_parser'
class DashboardController < ApplicationController

    before_action :authenticate_user!
    include HandParser

    def index
        @uploads_L = current_user.uploads.where("hand_pos == #{params[:hand_pos]} AND hand_LR == 'left'")
        @uploads_R = current_user.uploads.where("hand_pos == #{params[:hand_pos]} AND hand_LR == 'right'")
        @uploads_ns = current_user.uploads.where("hand_pos == #{params[:hand_pos]} AND hand_LR == 'not_selected'")
        hand_pos_name(params[:hand_pos])
        @archive = false
    end

    def archive
        @uploads_L = current_user.uploads.where("hand_pos == #{params[:hand_pos]} AND hand_LR == 'left'")
        @uploads_R = current_user.uploads.where("hand_pos == #{params[:hand_pos]} AND hand_LR == 'right'")
        @uploads_ns = current_user.uploads.where("hand_pos == #{params[:hand_pos]} AND hand_LR == 'not_selected'")
        hand_pos_name(params[:hand_pos])
        @archive = true
    end
    

end
