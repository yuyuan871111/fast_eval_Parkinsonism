module HandParser
    def hand_pos_name
        @hand_pos = "Please select the feature type."
        if params[:hand_pos].present? 
            # byebug
            if params[:hand_pos] == '1'
                @hand_pos = "Finger tapping"
            elsif params[:hand_pos] == '2'
                @hand_pos = "Open/close hand movement"
            elsif params[:hand_pos] == '3'
                @hand_pos = "Pronation/supination hand movement"
            end
        else
            @hand_pos = "Please select the feature type."
        end
    end
end
