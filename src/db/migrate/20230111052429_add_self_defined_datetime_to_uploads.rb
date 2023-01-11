class AddSelfDefinedDatetimeToUploads < ActiveRecord::Migration[7.0]
  def change
    add_column :uploads, :defined_time_at, :datetime
  end
end
