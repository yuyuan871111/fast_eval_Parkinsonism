class AddPredictedResultsToUploads < ActiveRecord::Migration[7.0]
  def change
    add_column :uploads, :updrs, :integer
    add_column :uploads, :err_frame_ratio, :float
    add_column :uploads, :mean_freq, :float
    add_column :uploads, :std_freq, :float
    add_column :uploads, :mean_inte, :float
    add_column :uploads, :std_inte, :float
    add_column :uploads, :mean_inte_freq, :float
    add_column :uploads, :std_inte_freq, :float
    add_column :uploads, :mean_peak, :float
    add_column :uploads, :std_peak, :float
  end
end
