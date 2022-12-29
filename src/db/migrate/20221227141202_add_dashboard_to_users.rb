class AddDashboardToUsers < ActiveRecord::Migration[7.0]
  def change
    create_table :dashboard do |t|
      t.datetime :created_at, precision: 6, null: false
      t.datetime :updated_at, precision: 6, null: false
      t.string :name, null: false
      t.string :state
      t.integer :file_size
      t.integer :user_id
    end
  end
end
