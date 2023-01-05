class RenameAvatarToResults < ActiveRecord::Migration[7.0]
  def change
    rename_column :uploads, :avatar, :results
  end
end
