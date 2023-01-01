module HandParser
    def hand_pos_name(params_hand_pos = nil)
        @hand_pos = "Please select the feature type."
        if params_hand_pos.present? 
            # byebug
            if params_hand_pos == '1'
                @hand_pos = "Finger tapping"
            elsif params_hand_pos == '2'
                @hand_pos = "Open/close hand movement"
            elsif params_hand_pos == '3'
                @hand_pos = "Pronation/supination hand movement"
            end
        else
            @hand_pos = "Please select the feature type."
        end
    end
end
