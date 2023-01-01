class AddStatusToUploads < ActiveRecord::Migration[7.0]
  def change
    remove_column :users, :user_id
    add_column :uploads, :status, :string
    drop_table :dashboard
  end
end
