class AddDataLimitToUsers < ActiveRecord::Migration[7.0]
  def change
    change_column :users, :data_usage, :integer
    add_column :users, :data_usage_limit, :integer
  end
end
