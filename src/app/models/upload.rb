class Upload < ApplicationRecord
    belongs_to :user
    after_initialize :init, unless: :persisted?
    
    # uploader
    # serialize :avatar, JSON # If you use SQLite, add this line.
    has_one_attached :video
    has_many_attached :results # where I store my results

    # hand_parameters
    validates :hand_pos, inclusion: { in: ['0', '1', '2', '3'] }
    validates :hand_LR, inclusion: { in: ['not_selected', 'left', 'right'] } 
    validates :status, inclusion: { in: ['uploaded', 'queuing', 'running', 'done', 'fail']}

    # file size limit
    validates :video, presence: true, blob: { content_type: :video, size_range: 1..(50.megabytes) }

    def init
        self.archived = false if self.archived.nil?
    end
end