rubyCopy# user_model.rb
class User < ApplicationRecord
  has_many :posts, dependent: :destroy
  has_one :profile
  
  validates :username, presence: true, uniqueness: true
  validates :email, presence: true, uniqueness: true
  
  def full_name
    "#{first_name} #{last_name}"
  end
  
  def self.active_users
    where(active: true)
  end
end