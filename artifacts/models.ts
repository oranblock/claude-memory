// models.ts
interface User {
  id: string;
  username: string;
  email: string;
  roles: UserRole[];
  createdAt: Date;
  isActive: boolean;
}

enum UserRole {
  ADMIN = 'admin',
  EDITOR = 'editor',
  VIEWER = 'viewer'
}

export type { User };
export { UserRole };