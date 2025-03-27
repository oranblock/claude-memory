// DataAccess.cs
using System;
using System.Data.SqlClient;
using System.Threading.Tasks;

namespace MyApp.Data
{
    public class UserRepository
    {
        private readonly string _connectionString;
        
        public UserRepository(string connectionString)
        {
            _connectionString = connectionString;
        }
        
        public async Task<User> GetUserByIdAsync(int userId)
        {
            using var connection = new SqlConnection(_connectionString);
            await connection.OpenAsync();
            // Implementation continues...
        }
    }
}