class AddDataUsageToUsers < ActiveRecord::Migration[7.0]
  def change
    remove_column :uploads, :file_size
    add_column :users, :data_usage, :float
  end
end
