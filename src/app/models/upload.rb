class Upload < ApplicationRecord
    belongs_to :user
    # uploader
    mount_uploader :avatar, AvatarUploader
    serialize :avatar, JSON # If you use SQLite, add this line.

end