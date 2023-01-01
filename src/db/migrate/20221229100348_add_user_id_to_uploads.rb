class AddUserIdToUploads < ActiveRecord::Migration[7.0]
  def change
    add_column :uploads, :user_id, :integer, unique: true
  end
end
