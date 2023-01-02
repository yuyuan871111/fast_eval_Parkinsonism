class AddRetrievedAtToUploads < ActiveRecord::Migration[7.0]
  def change
    add_column :uploads, :retrieved_at, :datetime
  end
end
