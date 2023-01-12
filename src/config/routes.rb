Rails.application.routes.draw do
  
  authenticate :user, lambda { |u| u.admin? } do
    # preview at http://localhost:3000/admin
    mount RailsAdmin::Engine => '/admin', as: 'rails_admin'
    
    # monitor jobs at http://localhost:3000/sidekiq
    require 'sidekiq/web'
    mount Sidekiq::Web => '/sidekiq'
  end

  devise_for :users, controllers: { 
    sessions: 'users/sessions', 
    registrations: 'users/registrations',
    passwords: 'users/passwords',
    confirmations: 'users/confirmations' 
  }

  devise_scope :user do  
    post '/users/sign_in' => 'users/sessions#new'
    get '/users/sign_out' => 'users/sessions#destroy'
    get '/users/sign_up' => 'users/registrations#new'
  end

  # preview at http://localhost:3000/letter_opener
  if Rails.env.development?
    mount LetterOpenerWeb::Engine, at: "/letter_opener"
  end
  # get 'home/index'
  # Define your application routes per the DSL in https://guides.rubyonrails.org/routing.html

  # Defines the root path route ("/")
  root "home#index", :as => 'root'
  get '/dashboard' => 'dashboard#index', :as => 'dashboard_index'
  get '/dashboard/archive' => 'dashboard#archive', :as => 'dashboard_archive'
  get '/documents' => 'home#documents', :as => 'documents'
  get '/demo' => 'home#demo', :as => 'demo'
  get '/status' => 'home#status', :as => 'status'
  
  resources :uploads
  get '/uploads/:id/do_run' => 'uploads#do_run', :as => 'do_run'
  get '/uploads/:id/archive'=> 'uploads#archive', :as => 'archive'
  get '/uploads/:id/unarchive'=> 'uploads#unarchive', :as => 'unarchive'

  # get '/test' => 'home#test', :as => 'test'

  
    
end
