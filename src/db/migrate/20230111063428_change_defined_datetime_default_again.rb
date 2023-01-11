class ChangeDefinedDatetimeDefaultAgain < ActiveRecord::Migration[7.0]
  def change
    change_column_default :uploads, :defined_time_at, DateTime.now
  end
end
