class AddUploadsToUsers < ActiveRecord::Migration[7.0]
  def change
    create_table :uploads do |t|
      t.datetime :created_at, precision: 6, null: false
      t.datetime :updated_at, precision: 6, null: false
      t.integer :file_size
    end
  end
end
