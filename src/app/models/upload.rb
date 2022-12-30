class Upload < ApplicationRecord
    belongs_to :user
    
    # uploader
    mount_uploader :avatar, AvatarUploader
    serialize :avatar, JSON # If you use SQLite, add this line.

    # hand_parameters
    validates :hand_pos, inclusion: { in: ['0', '1', '2', '3'] }
    validates :hand_LR, inclusion: { in: ['not_selected', 'left', 'right'] } 

end