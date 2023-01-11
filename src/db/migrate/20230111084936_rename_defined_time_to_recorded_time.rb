class RenameDefinedTimeToRecordedTime < ActiveRecord::Migration[7.0]
  def change
    rename_column :uploads, :defined_time_at, :recorded_at
    change_column :uploads, :recorded_at, :datetime, default: -> { 'CURRENT_TIMESTAMP' }
  end
end
