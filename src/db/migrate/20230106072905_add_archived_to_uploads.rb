class AddArchivedToUploads < ActiveRecord::Migration[7.0]
  def change
    add_column :uploads, :archived, :boolean
  end
end
