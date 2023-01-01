class AddAvatarToUploads < ActiveRecord::Migration[7.0]
  def change
    add_column :uploads, :avatar, :string
  end
end
