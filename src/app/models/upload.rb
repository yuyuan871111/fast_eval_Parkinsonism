class Upload < ApplicationRecord
    belongs_to :user
    
    # uploader
    # mount_uploader :avatar, AvatarUploader
    # serialize :avatar, JSON # If you use SQLite, add this line.
    has_one_attached :video
    has_many_attached :avatar # where I store my results

    # hand_parameters
    validates :hand_pos, inclusion: { in: ['0', '1', '2', '3'] }
    validates :hand_LR, inclusion: { in: ['not_selected', 'left', 'right'] } 
    validates :status, inclusion: { in: ['uploaded', 'queuing', 'running', 'done']}

    validates :video, presence: true, blob: { content_type: :video, size_range: 1..(20.megabytes) }
    # validates :photos, presence: true, blob: { content_type: %r{^image/}, size_range: 1..(5.megabytes) }
end