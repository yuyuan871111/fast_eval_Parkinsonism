class User < ApplicationRecord
  # User role
  enum role: [:user, :doctor, :admin]
  after_initialize :set_default_role, :if => :new_record?

  def set_default_role
    self.role ||= :user
    self.admin ||= false
    self.data_usage ||= 0
    self.data_usage_limit ||= 6442450944 # 6GB initial limit
  end

  # Set uploader
  has_many :uploads

  # Include default devise modules. Others available are:
  # :confirmable, :lockable, :timeoutable, :trackable and :omniauthable
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :validatable, :confirmable

end
