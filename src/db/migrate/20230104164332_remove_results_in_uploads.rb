class RemoveResultsInUploads < ActiveRecord::Migration[7.0]
  def change
    remove_column :uploads, :results
  end
end
