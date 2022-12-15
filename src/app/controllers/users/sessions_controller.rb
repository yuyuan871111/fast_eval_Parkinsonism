# frozen_string_literal: true

class Users::SessionsController < Devise::SessionsController
  # before_action :configure_sign_in_params, only: [:create]

  # GET /users/sign_in
  def new
    # super do |resource|
    #   BackgroundWorker.trigger(resource)
    # end
    super
  end

  # POST /users/sign_in
  # def create
  #   super
  # end

  # DELETE /users/sign_out
  def destroy
    super
  end

  protected

  def after_sign_in_path_for(resource)
    root_path
  end

  # If you have extra params to permit, append them to the sanitizer.
  # def configure_sign_in_params
  #   devise_parameter_sanitizer.permit(:sign_in, keys: [:attribute])
  # end
end
