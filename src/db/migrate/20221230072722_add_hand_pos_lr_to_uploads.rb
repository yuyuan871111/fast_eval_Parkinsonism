class AddHandPosLrToUploads < ActiveRecord::Migration[7.0]
  def change
    add_column :uploads, :hand_pos, :string
    add_column :uploads, :hand_LR, :string
  end
end
