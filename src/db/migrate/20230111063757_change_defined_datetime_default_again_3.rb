class ChangeDefinedDatetimeDefaultAgain3 < ActiveRecord::Migration[7.0]
  def change
    change_column :uploads, :defined_time_at, :datetime, default: -> { 'CURRENT_TIMESTAMP' }
  end
end
