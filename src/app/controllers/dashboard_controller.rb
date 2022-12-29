require 'hand_parser'
class DashboardController < ApplicationController

    before_action :authenticate_user!
    include HandParser
    def index
        @user_id = current_user.id
        hand_pos_name()
    end

end
