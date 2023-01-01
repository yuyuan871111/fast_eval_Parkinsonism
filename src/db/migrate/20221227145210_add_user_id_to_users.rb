class AddUserIdToUsers < ActiveRecord::Migration[7.0]
  def change
    add_column :users, :user_id, :integer, unique: true
  end
end
