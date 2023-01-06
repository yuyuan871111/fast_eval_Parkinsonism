class RenameRetrievedToSubmitted < ActiveRecord::Migration[7.0]
  def change
    rename_column :uploads, :retrieved_at, :submitted_at
  end
end
