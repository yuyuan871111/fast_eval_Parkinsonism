class AddUserReferenceToDashboard < ActiveRecord::Migration[7.0]
  def change
    remove_column :dashboard, :user_id
    add_reference :dashboard, :user, index: true
  end
end
