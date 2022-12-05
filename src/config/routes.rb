Rails.application.routes.draw do
  devise_for :users, controllers: { sessions: 'users/sessions' }

  devise_scope :user do  
    post '/users/sign_in' => 'users/sessions#create'
    get '/users/sign_out' => 'users/sessions#destroy'
  end
  #devise_scope :user do
  #  get 'sign_in', to: 'devise/sessions#new'
  #end

  # get 'home/index'
  # Define your application routes per the DSL in https://guides.rubyonrails.org/routing.html

  # Defines the root path route ("/")
  root "home#index", :as => 'root'
end
