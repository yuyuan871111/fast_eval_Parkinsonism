class ChangeDefinedDatetimeDefaultType < ActiveRecord::Migration[7.0]
  def change
    change_column_default :uploads, :defined_time_at, Time.current
  end
end
